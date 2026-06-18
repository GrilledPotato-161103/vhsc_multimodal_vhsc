import torch.nn as nn
from src.plugins.var import BreakpointContext, BreakpointOutput

class IdentityHook(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        return BreakpointOutput(fn_name=self.forward.__qualname__,
                                context=ctx,
                                output= ctx.inputs,
                                trace={"input": ctx.inputs})
