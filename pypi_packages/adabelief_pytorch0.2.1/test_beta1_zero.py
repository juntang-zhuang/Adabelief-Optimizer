"""beta1 = 0 must not send the second moment to eps."""
import torch

from adabelief_pytorch.AdaBelief import AdaBelief


def test_positive_beta1_still_uses_the_residual():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.tensor([1.0, -0.5]))
    opt = AdaBelief(
        [param],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        rectify=False,
        weight_decouple=False,
        print_change_log=False,
    )
    param.grad = torch.tensor([0.4, -0.2])
    opt.step()
    exp_avg = opt.state[param]["exp_avg"]
    # m = (1-0.9) g, residual = g - m, s = (1-0.999) residual^2 + eps
    residual = param.grad - exp_avg
    expected = (1 - 0.999) * residual * residual + 1e-8
    assert torch.allclose(opt.state[param]["exp_avg_var"], expected)


def test_beta1_zero_stays_finite_on_a_quadratic():
    param = torch.nn.Parameter(torch.tensor([0.5]))
    opt = AdaBelief(
        [param],
        lr=1e-2,
        betas=(0.0, 0.999),
        eps=1e-16,
        rectify=False,
        weight_decouple=False,
        weight_decay=0.0,
        print_change_log=False,
    )
    for _ in range(30):
        param.grad = (2 * param.detach()).clone()
        opt.step()
    assert torch.isfinite(param.detach()).all()
    assert param.detach().abs().item() < 0.5
