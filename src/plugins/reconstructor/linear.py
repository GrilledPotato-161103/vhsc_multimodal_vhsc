from typing import List, Tuple, Any
import math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import functional as F, init

from src.models.components.ffn import FeedForward
from src.plugins.var import BreakpointContext, BreakpointOutput

class HuberLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred, target):
        l1_norm = torch.abs(target - pred)
        if l1_norm < self.threshold:
            return 0.5 * (l1_norm ** 2).mean()
        else:
            return (self.threshold * (l1_norm - self.threshold)).mean()

class BilinearReconstructor(nn.Module): 
    def __init__(self,  d_1: int,
                        d_2: int,
                        hidden_dim: int = 32,
                        concat: bool = False,
                        activation: str = "relu",
                        norm: str = "none",
                        dropout: float = 0.3,
                        order: str = "adn",
                        dist: nn.Module = nn.L1Loss(reduction="none")):
        super().__init__()
        self.concat = concat
        self.d_1 = d_1
        self.d_2 = d_2
        self.ln12 = FeedForward(in_dim=d_1, hidden_dim=hidden_dim, out_dim=d_2, activation=activation, norm=norm, dropout=dropout, adn_order=order)
        self.ln21 = FeedForward(in_dim=d_2, hidden_dim=hidden_dim, out_dim=d_1, activation=activation, norm=norm, dropout=dropout, adn_order=order)
        self.dev1 = nn.Sequential(nn.Softplus(), FeedForward(in_dim=d_1 + d_2, hidden_dim=hidden_dim * 2, out_dim=d_1, activation=activation, norm=norm, dropout=dropout, adn_order=order))
        self.dev2 = nn.Sequential(nn.Softplus(), FeedForward(in_dim=d_1 + d_2, hidden_dim=hidden_dim * 2, out_dim=d_2, activation=activation, norm=norm, dropout=dropout, adn_order=order))        
        # To help on learning distance
        self.dist = dist
    
    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        # Expect what to do
        if not self.concat: 
            (mod_1, mod_2), kwargs = ctx.inputs, ctx.kwargs
        else:
            (latent,), kwargs = ctx.inputs, ctx.kwargs
            mod_1, mod_2 = latent[..., :self.d_1], latent[..., self.d_1:self.d_1 + self.d_2] 
        if ctx.bp_kwargs:
            (p1, p2) = ctx.bp_kwargs
        else:
            (p1, p2) = (1, 1)
        rec_2 = self.ln12(mod_1) if p1 else mod_2
        rec_1 = self.ln21(mod_2) if p2 else mod_1
        dist_1 = self.dist(rec_1, mod_2)
        dist_2 = self.dist(rec_2, mod_1)
        output = (rec_1, rec_2)
        if self.concat:
            merged = torch.cat(output, dim=-1)
            tail = latent[..., self.d_1 + self.d_2:]
            output = (torch.cat([merged, tail], dim=-1),) 
        dev_1 = self.dev1(torch.cat([rec_1, mod_2], dim=-1))
        dev_2 = self.dev2(torch.cat([rec_2, mod_1], dim=-1))
        cls = BreakpointOutput(
                fn_name=self.forward.__qualname__,
                context = ctx,
                output= output,
                trace={
                       "signal": ctx.bp_kwargs,
                       "input": (mod_1, mod_2),
                       "reconstructed": (rec_1, rec_2),
                       "distance": (dist_1, dist_2),
                       "dev": (dev_1, dev_2)}
        )

        return cls
    


class LinearReconstructor(nn.Module):
    counter = 0
    def __init__(self, 
                    modals: List[nn.Module],
                    dim: int,
                    id: str | None = None):
        super().__init__()
        self.modals = dict()
        self.register(modals)
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=True) for i in range(len(modals))])
        self.data = dict()
        self.proj_var = torch.Parameter(0)
        self.id = id if isinstance(id) else f"recon.{LinearReconstructor.counter}"
        LinearReconstructor.counter += 1
        
    def collect(self, modal_id, input=None):
        if modal_id not in self.modals.keys():
            raise ValueError("Wrong hook id")
        self.data[modal_id] = self.proj[self.modals[modal_id]](input) if input is not None else input
    
    def is_ready(self): 
        return len(self.data) == len(self.modals)

    def forward(self, id, input=None):
        # Semaphore to interrupt model inference.
        self.collect(self, id, input)
        while not self.is_ready():
            pass
        def inv(x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
            bias = layer.bias if layer.bias is not None else 0
            return torch.linalg.solve(layer.weight, x - bias)
        
        self.proj_loss.zero_grad()
        # Reconstruct missing modality by inverse transform
        for rec_id in self.data.keys():
            if self.data[rec_id] is not None:
                rec_tensor.append(inv(self.data[rec_id], self.proj[id]))
        rec_tensor = torch.stack(rec_tensor, dim=1)
        # Penalty on projection variance
        self.proj_var += torch.var(rec_tensor, dim=1).mean()
        # Reconstruct by mean of projections
        return rec_tensor.mean(dim=1)

    # Register hook 
    def register(self, modals): 
        for modal in modals:
            self.modals[modal.id] = modal

if __name__ == "__main__":
    pass 

