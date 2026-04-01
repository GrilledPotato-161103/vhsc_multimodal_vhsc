"""
This file contains only variable and dataclass 
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Iterable, Tuple, Union, Literal, List
import torch
from torch import nn


def _indent(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.splitlines())

def _format_tensor(x: torch.Tensor) -> str:
    return (
        f"Tensor(shape={tuple(x.shape)}, "
        f"dtype={x.dtype}, "
        f"device={x.device}, "
        f"requires_grad={x.requires_grad})"
    )


def _format_value(value: Any, indent: int = 0) -> str:
    space = " " * indent

    if value is None:
        return "None"

    if isinstance(value, torch.Tensor):
        return _format_tensor(value)

    if isinstance(value, nn.Module):
        # Use builtin print representation of nn.Module
        return f"\n{_indent(str(value), indent + 2)}"

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            formatted_v = _format_value(v, indent + 2)
            if "\n" in formatted_v:
                lines.append(f"{space}{k}:")
                lines.append(_indent(formatted_v.strip("\n"), 2))
            else:
                lines.append(f"{space}{k}: {formatted_v}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            formatted_item = _format_value(item, indent + 2)
            if "\n" in formatted_item:
                lines.append(f"{space}-")
                lines.append(_indent(formatted_item.strip("\n"), 2))
            else:
                lines.append(f"{space}- {formatted_item}")
        return "\n".join(lines)

    if isinstance(value, tuple):
        if not value:
            return "()"
        lines = []
        for i, item in enumerate(value):
            formatted_item = _format_value(item, indent + 2)
            if "\n" in formatted_item:
                lines.append(f"{space}[{i}]:")
                lines.append(_indent(formatted_item.strip("\n"), 2))
            else:
                lines.append(f"{space}[{i}]: {formatted_item}")
        return "\n".join(lines)

    return repr(value)


def _format_dataclass(obj: Any, title: str, fields: Dict[str, Any]) -> str:
    lines = [f"{title}("]
    for key, value in fields.items():
        formatted = _format_value(value, indent=2)
        if "\n" in formatted:
            lines.append(f"  {key}=")
            lines.append(_indent(formatted.strip("\n"), 4))
        else:
            lines.append(f"  {key}={formatted}")
    lines.append(")")
    return "\n".join(lines)


# To store Breakpoint runtime context
@dataclass
class BreakpointContext:
    name: str
    layer: str
    position: str   # "before" or "after"
    module: nn.Module
    inputs: tuple
    bp_kwargs: dict
    kwargs: dict
    output: Any = None
    state: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return _format_dataclass(
            self,
            "BreakpointContext",
            {
                "name": self.name,
                "layer": self.layer,
                "position": self.position,
                "module": self.module,
                "inputs": self.inputs,
                "bp_kwargs": self.bp_kwargs,
                "kwargs": self.kwargs,
                "output": self.output,
                "state": self.state,
            },
        )

    __repr__ = __str__


# To store Breakpoint Output
@dataclass
class BreakpointOutput:    
    fn_name: str
    output: Dict | List[torch.Tensor] | torch.Tensor | None = None
    trace: Dict | List[torch.Tensor] | torch.Tensor | None = None
    valid: bool = False 

    def __str__(self) -> str:
        return _format_dataclass(
            self,
            "BreakpointOutput",
            {
                "fn_name": self.fn_name,
                "output": self.output,
                "trace": self.trace,
                "valid": self.valid
            },
        )

    __repr__ = __str__