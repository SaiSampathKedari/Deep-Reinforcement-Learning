import torch
import torch.nn as nn

def layer_init(
  layer     :   nn.Linear,
  std       :   float = 2**0.5,
  bias_const:   float = 0.0,  
) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(
    sizes       :   list[int],
    output_gain :   float,
    activation  :   type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    
    layers: list[nn.Module] = []
    for i in range(len(sizes) -1):
        last = i == len(sizes) - 2
        gain = output_gain if last else 2**0.5
        layers.append(
            layer_init(nn.Linear(sizes[i], sizes[i+1]), std=gain)
        )
        if not last:
            layers.append(activation())
    return nn.Sequential(*layers)

def grad_norm(module : nn.Module) -> float:
    
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack([g.norm(2) for g in grads]).norm(2))

@torch.no_grad()
def explained_variance(
    y_pred : torch.Tensor,
    y_true: torch.Tensor
) -> float:
    var_true = torch.var(y_true, unbiased=False)
    if var_true ==0:
        return float("nan")
    return float(1 - torch.var(y_true - y_pred, unbiased=False) / var_true)