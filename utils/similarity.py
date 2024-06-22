from gauss_model import GaussOutput
import torch
from torch import distributions

def asymmetrical_kl_sim(mu1: torch.FloatTensor, std1: torch.FloatTensor, mu2: torch.FloatTensor, std2: torch.FloatTensor) -> torch.Tensor:
    p1 = distributions.normal.Normal(mu1, std1)
    p2 = distributions.normal.Normal(mu2, std2)
    sim = 1 / (1 + distributions.kl.kl_divergence(p1, p2))

    return sim.mean(dim=-1)

def asymmetrical_kl_sim_mat(mu1: torch.FloatTensor, std1: torch.FloatTensor, mu2: torch.FloatTensor, std2: torch.FloatTensor) -> torch.Tensor:
    p1 = distributions.normal.Normal(mu1.unsqueeze(0), std1.unsqueeze(0))
    p2 = distributions.normal.Normal(mu2.unsqueeze(1), std2.unsqueeze(1))
    sim = 1 / (1 + distributions.kl.kl_divergence(p1, p2))

    return sim.mean(dim=-1)