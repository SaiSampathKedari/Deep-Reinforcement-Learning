"""Redirect the retired demo command to its repository example."""


def main() -> None:
    raise SystemExit(
        "The deeprl.train demo command is retired. From the repository root, run:\n"
        "  python -m examples.gymnasium.cartpole.run_cartpole_one_step_actor_critic "
        "--output-dir outputs/gymnasium/cartpole/one_step_actor_critic/run_01\n"
        "Add --no-video to skip recording."
    )


if __name__ == "__main__":
    main()
