"""Tests for deeprl.logger.Logger.

These check the four behaviours an algorithm silently depends on: episode
accounting, independence between environments, interval averaging, and the
final flush. Each one, if broken, produces plausible-looking numbers rather
than an error -- which is why they are asserted rather than eyeballed.
"""

import math

import pytest
import torch

from deeprl.logger import Logger


def _values(history, name):
    """Just the values of a metric, dropping the step column."""
    return [v for _, v in history[name]]


def _steps(history, name):
    return [s for s, _ in history[name]]


# --------------------------------------------------------------------------
# 1. Episode tracking
# --------------------------------------------------------------------------


def test_episode_return_and_length_match_a_manual_sum():
    """A finished episode reports exactly the rewards it accumulated."""
    logger = Logger(num_envs=1, log_every=0)
    rewards = [1.0, 2.0, 3.0, 4.0]

    for i, r in enumerate(rewards):
        done = torch.tensor([i == len(rewards) - 1])
        logger.log_step(torch.tensor([r]), done)

    history = logger.finish()
    assert _values(history, "charts/episodic_return") == [sum(rewards)]
    assert _values(history, "charts/episodic_length") == [len(rewards)]


def test_counters_reset_between_episodes():
    """The second episode does not inherit the first one's total."""
    logger = Logger(num_envs=1, log_every=0)
    for r, d in [(1.0, False), (2.0, True), (5.0, False), (6.0, True)]:
        logger.log_step(torch.tensor([r]), torch.tensor([d]))

    history = logger.finish()
    assert _values(history, "charts/episodic_return") == [3.0, 11.0]
    assert _values(history, "charts/episodic_length") == [2.0, 2.0]


def test_global_step_advances_by_num_envs():
    logger = Logger(num_envs=4, log_every=0)
    for _ in range(10):
        logger.log_step(torch.zeros(4), torch.zeros(4, dtype=torch.bool))
    assert logger.global_step == 40


# --------------------------------------------------------------------------
# 2. Asynchronous completion
# --------------------------------------------------------------------------


def test_finishing_one_env_does_not_disturb_the_others():
    """The bug that makes RecordEpisodeStatistics unusable with reset_mask.

    env 0 finishes on step 2; env 1 keeps running to step 4. If env 0's reset
    also cleared env 1, env 1 would report 2.0 instead of 4.0.
    """
    logger = Logger(num_envs=2, log_every=0)
    dones = [
        [False, False],
        [True, False],   # env 0 ends after 2 steps
        [False, False],
        [False, True],   # env 1 ends after 4 steps
    ]
    for step_dones in dones:
        logger.log_step(torch.ones(2), torch.tensor(step_dones))

    history = logger.finish()
    assert _values(history, "charts/episodic_return") == [2.0, 4.0]
    assert _values(history, "charts/episodic_length") == [2.0, 4.0]


def test_simultaneous_completions_are_recorded_separately():
    """Two envs finishing on the same step produce two entries, not one."""
    logger = Logger(num_envs=3, log_every=0)
    logger.log_step(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, False, False]))
    logger.log_step(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False, True]))

    history = logger.finish()
    assert sorted(_values(history, "charts/episodic_return")) == [2.0, 6.0]
    assert _steps(history, "charts/episodic_return") == [6, 6]  # same global step


# --------------------------------------------------------------------------
# 3. Interval averaging
# --------------------------------------------------------------------------


def test_metrics_are_averaged_over_the_interval():
    """Four known values in one interval come out as their mean."""
    logger = Logger(num_envs=1, log_every=4)
    for value in (1.0, 2.0, 3.0, 4.0):
        logger.log_step(
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            metrics={"losses/x": value},
        )

    history = logger.finish()
    assert _values(history, "losses/x") == [2.5]
    assert _steps(history, "losses/x") == [4]


def test_each_metric_is_averaged_by_its_own_count():
    """A metric recorded on only some steps is not diluted by the others.

    `sometimes` is recorded twice with value 10; `always` four times with 1.
    A single shared counter would report sometimes = 20/4 = 5.
    """
    logger = Logger(num_envs=1, log_every=4)
    for i in range(4):
        metrics = {"losses/always": 1.0}
        if i % 2 == 0:
            metrics["losses/sometimes"] = 10.0
        logger.log_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool), metrics=metrics)

    history = logger.finish()
    assert _values(history, "losses/always") == [1.0]
    assert _values(history, "losses/sometimes") == [10.0]


def test_interval_boundaries_do_not_drift_with_num_envs():
    """global_step jumps by num_envs and may never equal a multiple of log_every."""
    logger = Logger(num_envs=3, log_every=10)
    for _ in range(10):  # global_step: 3, 6, 9, 12, ... 30
        logger.log_step(torch.zeros(3), torch.zeros(3, dtype=torch.bool),
                        metrics={"losses/x": 1.0})

    history = logger.finish()
    assert _steps(history, "losses/x") == [12, 21, 30]


# --------------------------------------------------------------------------
# 4. Final flush
# --------------------------------------------------------------------------


def test_finish_flushes_an_incomplete_interval():
    """Two steps into a 100-step interval, finish() still records them."""
    logger = Logger(num_envs=1, log_every=100)
    for _ in range(2):
        logger.log_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                        metrics={"losses/x": 7.0})

    history = logger.finish()
    assert _values(history, "losses/x") == [7.0]
    assert _steps(history, "losses/x") == [2]


def test_finish_flushes_explained_variance_without_scalar_metrics():
    """EV alone is enough to justify a final flush."""
    logger = Logger(num_envs=2, log_every=100, ev_name="diagnostics/ev")
    torch.manual_seed(0)
    for _ in range(5):
        targets = torch.randn(2)
        logger.log_step(torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                        values=targets, targets=targets)

    history = logger.finish()
    assert history["diagnostics/ev"][-1][1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_explained_variance_uses_the_configured_name():
    logger = Logger(num_envs=1, log_every=1, ev_name="diagnostics/td_target_ev")
    logger.log_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                    values=torch.tensor([0.5]), targets=torch.tensor([1.0]))
    assert "diagnostics/td_target_ev" in logger.history


def test_wrong_shapes_raise():
    logger = Logger(num_envs=4, log_every=0)
    with pytest.raises(ValueError, match="rewards must have shape"):
        logger.log_step(torch.zeros(3), torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="dones must have shape"):
        logger.log_step(torch.zeros(4), torch.zeros(3, dtype=torch.bool))


def test_non_scalar_metric_raises():
    logger = Logger(num_envs=1, log_every=1)
    with pytest.raises(ValueError, match="only scalar metrics"):
        logger.log_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                        metrics={"losses/x": torch.zeros(5)})


def test_invalid_construction_raises():
    with pytest.raises(ValueError, match="num_envs must be positive"):
        Logger(num_envs=0)
    with pytest.raises(ValueError, match="log_every must be non-negative"):
        Logger(num_envs=1, log_every=-1)


def test_history_is_a_plain_dict_of_step_value_pairs():
    logger = Logger(num_envs=1, log_every=0)
    logger.log_step(torch.tensor([1.0]), torch.tensor([True]))
    history = logger.finish()

    assert type(history) is dict
    assert history["charts/episodic_return"] == [(1, 1.0)]
    assert not math.isnan(logger.recent("charts/episodic_return"))
    assert math.isnan(logger.recent("does/not/exist"))
