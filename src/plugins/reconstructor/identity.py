from typing import List, Tuple, Any
import math
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from src.plugins.var import BreakpointContext, BreakpointOutput

class IdentityHook(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        return BreakpointOutput(fn_name=self.forward.__qualname__,
                                context=ctx,
                                output= ctx.inputs,
                                trace={"input": ctx.inputs})
