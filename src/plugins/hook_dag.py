from __future__ import annotations
import pickle
import torch
import torch.nn as nn
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Optional, Any, Callable, Union, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from src.plugins.var import BreakpointContext, BreakpointOutput, _format_dataclass

# Module-level registry — survives module reimports that would otherwise
# create a fresh defaultdict if it were a class attribute.
# _breakpoint_registry: Dict[str, List["Breakpoint"]] = 


class Breakpoint(nn.Module):
    # Backward-compatible alias; internal code should reference
    # _breakpoint_registry directly for clarity.
    list_of_breakpoints: Dict[str, List["Breakpoint"]] = defaultdict(list)

    def __init__(
        self,
        name: str,
        callback: Optional[Callable[[BreakpointContext], Any]] = None,
        mutate: bool = False,
        valid: bool = False,
        kwargs: dict = dict(),
        data_sources: List[str] | None = None,
        pre_fn: Optional[Callable[[BreakpointContext], Any]] = None,
        post_fn: Optional[Callable[[BreakpointContext], Any]] = None,
    ):
        super().__init__()
        self.callback = callback
        self.mutate = mutate
        self.valid = valid
        self.trace = None
        self.kwargs = kwargs
        # DAG wiring: user-declared upstream breakpoint names.
        # The special value 'input' means "include the model's own
        # captured data from the hook" (ctx.inputs for before,
        # ctx.output for after).
        self.data_sources: List[str] = list(data_sources) if data_sources else []
        # DAG wiring: downstream breakpoints (resolved by controller.wire())
        self.data_sinks: List[Breakpoint] = []
        # Runtime buffer for data pushed from upstream breakpoints
        self._buffer: Dict[str, Any] = {}
        # Optional input-preparation callable.  When set, pre_fn(ctx) is
        # called before callback(ctx) and its return value is passed to
        # the callback instead of the raw context.
        self.pre_fn: Optional[Callable[[BreakpointContext], Any]] = pre_fn
        self.post_fn: Optional[Callable[[BreakpointContext], Any]] = post_fn
        Breakpoint.list_of_breakpoints[name].append(self)
        self.name = f"{name}.{len(Breakpoint.list_of_breakpoints[name]) - 1}"
        print(f"Added bp: {self.name} to breakpoints list")

    def reset(self) -> None:
        """Clear runtime buffers between forward passes."""
        self._buffer.clear()
        self.trace = None
    
    def __str__(self):
        return _format_dataclass(
            self,
            "Breakpoint",
            {   
                "name": self.name,
                "callback": self.callback,
                "mutate": self.mutate, 
                "valid": self.valid,
                "kwargs": self.kwargs,
            },
        )

    @staticmethod
    def get_by_name(query):
        keys = query.split(".")
        cur = Breakpoint.list_of_breakpoints
        for k in keys:
            # try index if it's a number
            if k.isdigit():
                cur = cur[int(k)]
            else:
                cur = cur[k]
        return cur

    def run_before(
        self,
        layer_name: str,
        module: nn.Module,
        inputs: tuple,
        kwargs: dict,
        state=None,
    ):
        # Source breakpoint (no callback): push raw hook data to sinks
        if self.callback is None:
            for sink in self.data_sinks:
                sink._buffer[self.name] = inputs
            return inputs, kwargs

        collected = dict(self._buffer)  # snapshot of upstream data

        # Include model's own captured data if 'input' is in data_sources
        if "input" in self.data_sources:
            collected["input"] = inputs

        ctx = BreakpointContext(
            name=self.name,
            layer=layer_name,
            position="before",
            module=module,
            inputs=inputs,
            kwargs=kwargs,
            bp_kwargs=self.kwargs,
            state=state,
            collected=collected,
        )

        # pre_fn transforms the context before the callback sees it
        callback_input = self.pre_fn(ctx) if self.pre_fn is not None else ctx
        result = self.callback(callback_input)
        
        if self.post_fn is not None:
            result = self.post_fn(result)
        self.trace = result

        # Push result to downstream breakpoints (DAG data flow)
        for sink in self.data_sinks:
            sink._buffer[self.name] = result.output
        new_inputs = result.output
        if self.mutate and new_inputs is not None:
            if isinstance(new_inputs, tuple):
                return new_inputs, kwargs
            raise ValueError("Before-breakpoint callback must return (inputs, kwargs)")
        return inputs, kwargs

    def run_after(
        self,
        layer_name: str,
        module: nn.Module,
        inputs: tuple,
        kwargs: dict,
        output: Any,
        state=None,
    ):
        # Source breakpoint (no callback): push raw hook data to sinks
        if self.callback is None:
            for sink in self.data_sinks:
                sink._buffer[self.name] = output
            return output

        collected = dict(self._buffer)  # snapshot of upstream data

        # Include model's own captured data if 'input' is in data_sources
        if "input" in self.data_sources:
            collected["input"] = output

        ctx = BreakpointContext(
            name=self.name,
            layer=layer_name,
            position="after",
            module=module,
            inputs=inputs,
            kwargs=kwargs,
            bp_kwargs=self.kwargs,
            output=output,
            state=state,
            collected=collected,
        )

        # pre_fn transforms the context before the callback sees it
        callback_input = self.pre_fn(ctx) if self.pre_fn is not None else ctx
        result = self.callback(callback_input)
        self.trace = result
        # Push result to downstream breakpoints (DAG data flow)
        for sink in self.data_sinks:
            sink._buffer[self.name] = result.output
        if self.mutate and result.output is not None:
            return result.output
        return output


class BreakpointController:

    def __init__(self):
        self.breakpoints: List[Dict[str, Any]] = []
        self.handles: List[Any] = []
        self.state: Dict[str, Any] = {}
        self._wired: bool = False

    @staticmethod
    def __init_dict__(model: nn.Module, cfg: DictConfig) -> BreakpointController:
        controller = BreakpointController()
        assert type(model).__name__ == cfg.target, "Plugins are going to be plugged into wrong model."
        for item in cfg.breakpoints:
            bp = instantiate(item.bp)
            controller.add_breakpoint_by_name(model, item.layer_name, bp, item.pos)
        controller.wire()
        return controller

    @staticmethod
    def load_from_state_dict(
        root: nn.Module,
        data: dict,
        strict: bool = True
    ):
        controller = BreakpointController()
        controller.state = data.get("state", {})
        loaded = []
        skipped = []

        for spec in data.get("breakpoints", []):
            callback = spec.get("callback")

            if callback is None:
                msg = f"Callback '{callback}' is not registered."
                if strict:
                    raise ValueError(msg)
                skipped.append({"spec": spec, "reason": msg})
                continue

            bp = Breakpoint(
                name=spec["name"],
                callback=callback,
                mutate=spec.get("mutate", False),
                valid=spec.get("valid", False),
                kwargs=spec.get("kwargs", {}),
                data_sources=spec.get("data_sources", []),
            )
            try:
                controller.add_breakpoint(
                    root=root,
                    target=spec["layer_name"],
                    bp=bp,
                    position=spec.get("position", "after"),
                )
                loaded.append(spec)
            except Exception as e:
                if strict:
                    raise
                skipped.append({"spec": spec, "reason": str(e)})

        controller.wire()
        return controller, {
                                "loaded": loaded,
                                "skipped": skipped,
                            }
    @staticmethod
    def load_from_checkpoint(
        root: nn.Module,
        path: str,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load breakpoint configuration and re-attach hooks to `root`.

        strict=True:
            raise error if a layer name or callback cannot be resolved.
        strict=False:
            skip unresolved entries.
        """
        
        data = torch.load(path, map_location="cpu", weights_only=False)
        return BreakpointController.load_from_state_dict(root, data, strict)
    
    @staticmethod
    def _named_modules_map(root: nn.Module) -> Dict[str, nn.Module]:
        return dict(root.named_modules())

    @staticmethod
    def _module_to_names(root: nn.Module) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = {}
        for name, mod in root.named_modules():
            out.setdefault(id(mod), []).append(name)
        return out

    def _resolve_target(
        self,
        root: nn.Module,
        target: Union[str, nn.Module],
    ) -> Tuple[str, nn.Module]:
        """
        Resolve a breakpoint target into (layer_name, module).

        target can be:
        - str: module path from named_modules(), e.g. "layer1.0.conv1"
        - nn.Module: actual module object from inside root
        """
        name_to_module = self._named_modules_map(root)

        if isinstance(target, str):
            if target not in name_to_module:
                available = ", ".join(list(name_to_module.keys())[:20])
                raise ValueError(
                    f"Layer name '{target}' not found in model.named_modules(). "
                    f"Available examples: {available}"
                )
            return target, name_to_module[target]

        if isinstance(target, nn.Module):
            module_to_names = self._module_to_names(root)
            names = module_to_names.get(id(target), None)
            if not names:
                raise ValueError("Target module is not a submodule of the provided root model.")

            # Prefer the shortest non-empty name if possible.
            chosen_name = sorted(names, key=lambda x: (x == "", len(x), x))[0]
            return chosen_name, target

        raise TypeError("target must be either a layer name (str) or an nn.Module")

    def gather(self):
        return [
            (bp_info["breakpoint"], (bp_info["breakpoint"].trace.output, bp_info["breakpoint"].trace.valid))
            for bp_info in self.breakpoints
        ]

    def add_breakpoint(
        self,
        root: nn.Module,
        target: Union[str, nn.Module],
        bp: Breakpoint,
        position: str = "after",
    ):
        """
        Add breakpoint to a target module specified by:
        - target='layer1.0.conv1'
        - target=model.layer1[0].conv1
        """
        layer_name, module = self._resolve_target(root, target)

        if position == "before":
            def hook(mod, args, kwargs):
                new_args, new_kwargs = bp.run_before(
                    layer_name=layer_name,
                    module=mod,
                    inputs=args,
                    kwargs=kwargs,
                    state=self.state,
                )
                return new_args, new_kwargs

            handle = module.register_forward_pre_hook(hook, with_kwargs=True)

        elif position == "after":
            def hook(mod, args, kwargs, output):
                return bp.run_after(
                    layer_name=layer_name,
                    module=mod,
                    inputs=args,
                    kwargs=kwargs,
                    output=output,
                    state=self.state,
                )

            handle = module.register_forward_hook(hook, with_kwargs=True)

        else:
            raise ValueError("position must be 'before' or 'after'")

        self.breakpoints.append(
            {
                "layer_name": layer_name,
                "module": module,
                "position": position,
                "breakpoint": bp,
                "handle": handle,
            }
        )
        self.handles.append(handle)
        self._wired = False  # adding a breakpoint invalidates wiring
        return handle

    # ------------------------------------------------------------------
    # DAG wiring
    # ------------------------------------------------------------------

    def wire(self) -> None:
        """Resolve all ``data_sources`` names to ``data_sinks`` references.

        Should be called after all breakpoints have been added to the
        controller.  Skips if already wired (idempotent, reset by
        :meth:`add_breakpoint`).

        Also runs cycle detection via :meth:`_validate_dag`.
        """
        if self._wired:
            return

        # Clear existing wiring (supports re-wire after add_breakpoint)
        for item in self.breakpoints:
            item["breakpoint"].data_sinks.clear()

        # Build name → Breakpoint lookup
        bp_by_name: Dict[str, Breakpoint] = {}
        for item in self.breakpoints:
            bp: Breakpoint = item["breakpoint"]
            bp_by_name[bp.name] = bp

        # Resolve each breakpoint's data_sources → populate sinks
        for item in self.breakpoints:
            bp: Breakpoint = item["breakpoint"]
            for src_name in bp.data_sources:
                # 'input' is a runtime keyword resolved from hook data, not a breakpoint
                if src_name == "input":
                    continue
                upstream = bp_by_name.get(src_name)
                if upstream is None:
                    # Try base-name fallback via global registry
                    parts = src_name.split(".")
                    base = parts[0]
                    if base in Breakpoint.list_of_breakpoints:
                        if len(parts) == 1:
                            upstream = Breakpoint.list_of_breakpoints[base][-1]
                        else:
                            try:
                                idx = int(parts[1])
                                upstream = Breakpoint.list_of_breakpoints[base][idx]
                            except (IndexError, ValueError):
                                pass
                if upstream is None:
                    raise ValueError(
                        f"Breakpoint '{bp.name}' declares data_source "
                        f"'{src_name}' which does not match any registered "
                        f"breakpoint. Available: {list(bp_by_name.keys())}"
                    )
                upstream.data_sinks.append(bp)

        # Validate DAG (cycle detection)
        self._validate_dag()
        self._wired = True

    def _validate_dag(self) -> None:
        """DFS tricolour cycle detection over ``data_sources``."""
        # Build adjacency: breakpoint name → list of upstream source names
        adj: Dict[str, List[str]] = {}
        for item in self.breakpoints:
            bp = item["breakpoint"]
            adj[bp.name] = list(bp.data_sources)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {name: WHITE for name in adj}

        def dfs(name: str) -> None:
            color[name] = GRAY
            for src_name in adj.get(name, []):
                if src_name == "input" or src_name not in color:
                    continue  # 'input' is runtime, not a breakpoint
                if color[src_name] == GRAY:
                    raise ValueError(
                        f"Cycle detected in breakpoint DAG: "
                        f"'{name}' depends on '{src_name}' (back edge)"
                    )
                if color[src_name] == WHITE:
                    dfs(src_name)
            color[name] = BLACK

        for name in color:
            if color[name] == WHITE:
                dfs(name)

    def add_breakpoint_by_name(
        self,
        root: nn.Module,
        layer_name: str,
        bp: Breakpoint,
        position: str = "after",
    ):
        return self.add_breakpoint(root=root, target=layer_name, bp=bp, position=position)

    def eval(self):
        for item in self.breakpoints:
            if isinstance(item["breakpoint"].callback, nn.Module):
                item["breakpoint"].callback.eval()
    
    def train(self):
        for item in self.breakpoints:
            if isinstance(item["breakpoint"].callback, nn.Module):
                item["breakpoint"].callback.train()
    
    def to(self, device:str):
        for item in self.breakpoints:
            if isinstance(item["breakpoint"].callback, nn.Module):
                item["breakpoint"].callback.to(device)

    def cuda(self):
        for item in self.breakpoints:
            if isinstance(item["breakpoint"].callback, nn.Module):
                item["breakpoint"].callback.cuda()
    

    def add_breakpoint_by_module(
        self,
        root: nn.Module,
        module: nn.Module,
        bp: Breakpoint,
        position: str = "after",
    ):
        return self.add_breakpoint(root=root, target=module, bp=bp, position=position)

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        return [
            {
                "layer_name": item["layer_name"],
                "position": item["position"],
                "breakpoint_name": item["breakpoint"].name,
                "module_type": type(item["module"]).__name__,
                "mutate": item["breakpoint"].mutate,
                "valid": item["breakpoint"].valid,
                "callback": item["breakpoint"].callback,
                "kwargs": item["kwargs"].kwargs
            }
            for item in self.breakpoints
        ]

    def state_dict(self) -> Dict[str, Any]:
        """
        Export only serializable breakpoint configuration.
        Does not export raw module objects or hook handles.
        """
        return {
            "state": self.state,
            "breakpoints": [
                {
                    "name": item["breakpoint"].name,
                    "layer_name": item["layer_name"],
                    "position": item["position"],
                    "mutate": item["breakpoint"].mutate,
                    "callback": item["breakpoint"].callback,
                    "data_sources": item["breakpoint"].data_sources,
                }
                for item in self.breakpoints
            ]
        }

    def save(self, path: str, use_torch: bool = True):
        data = self.state_dict()
        if use_torch:
            torch.save(data, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def load(
        self,
        root: nn.Module,
        path: str,
        use_torch: bool = True,
        clear_existing: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load breakpoint configuration and re-attach hooks to `root`.

        strict=True:
            raise error if a layer name or callback cannot be resolved.
        strict=False:
            skip unresolved entries.
        """
        if use_torch:
            data = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        if clear_existing:
            self.clear()

        self.state = data.get("state", {})

        loaded = []
        skipped = []

        for spec in data.get("breakpoints", []):
            callback = spec.get("callback", None)

            if callback is None:
                msg = f"Callback key '{callback}' is not registered."
                if strict:
                    raise ValueError(msg)
                skipped.append({"spec": spec, "reason": msg})
                continue

            bp = Breakpoint(
                name=spec["name"],
                callback=callback,
                mutate=spec.get("mutate", False),
                valid=spec.get("valid", False),
                kwargs=spec.get("kwargs", {}),
                data_sources=spec.get("data_sources", []),
            )

            try:
                self.add_breakpoint(
                    root=root,
                    target=spec["layer_name"],
                    bp=bp,
                    position=spec.get("position", "after"),
                )
                loaded.append(spec)
            except Exception as e:
                if strict:
                    raise
                skipped.append({"spec": spec, "reason": str(e)})

        self.wire()
        return {
            "loaded": loaded,
            "skipped": skipped,
        }

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for item in self.breakpoints:
            item["breakpoint"].reset()
        self.breakpoints.clear()

# ---------------------------------------------------------------------------
# Built-in pre_fn utilities
# ---------------------------------------------------------------------------

class PreprocessCollectedFn(nn.Module):
    # A general preprocess function for collected contexts
    def __init__(self, fns: Dict[Callable]): 
        super().__init__()
        self.fns = fns
    
    def forward(self, ctx: BreakpointContext) -> BreakpointContext: 
        collected = ctx.collected
        for key in self.fns.keys():
            entries = [item[key] for item in collected]
            setattr(ctx, key, self.fns[key](entries))
        return ctx

class ConcatCollectedFn(nn.Module):
    """``pre_fn`` that concatenates all tensors in ``ctx.collected``.

    Packs the collected data into ``ctx.inputs`` so that existing callbacks
    (e.g. :class:`~src.plugins.reconstructor.linear.BilinearReconstructor`)
    that read ``ctx.inputs`` can consume DAG-routed data without changes.

    Keys in ``collected`` are sorted alphabetically for deterministic order.
    """

    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        tensors = [
            v for _, v in sorted(ctx.collected.items())
            if isinstance(v, torch.Tensor)
        ]
        if tensors:
            ctx.inputs = (torch.cat(tensors, dim=-1),)
        return ctx

class ToListCollectedFn(nn.Module):
    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        # print(ctx.collected)
        tensors = [
            v for _, v in sorted(ctx.collected.items())
            if isinstance(v, torch.Tensor)
        ]
        if tensors:
            ctx.inputs = tuple(tensors)
        return ctx

class SumCollectedFn(nn.Module):
    """``pre_fn`` that element-wise sums all tensors in ``ctx.collected``.

    Useful for :class:`~src.models.components.toy.MultiModalRegressor`
    whose ``head`` expects a single summed latent vector.
    """

    def forward(self, ctx: BreakpointContext) -> BreakpointContext:
        tensors = [
            v for v in ctx.collected.values()
            if isinstance(v, torch.Tensor)
        ]
        if tensors:
            ctx.inputs = (torch.stack(tensors).sum(dim=0),)
        return ctx

class SumPostOp(nn.Module): 
    def forward(self, res: BreakpointOutput) -> BreakpointOutput:
        tensors = res.output
        res.output = (torch.sum(torch.stack(tensors, dim=0), dim=0), )
        return res

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
    def main(cfg: DictConfig) -> Optional[float]:
        # print(cfg)
        plugin_cfg = cfg.plugins
        print("Initializing model")
        model = torch.load(plugin_cfg.model_checkpoint, weights_only=False).cuda()
        model.requires_grad_(False)
        datamodule = instantiate(cfg.data)
        # print(type(datamodule)
        datamodule.setup()
        loader = datamodule.val_dataloader()
        data = iter(loader)
        (x1, x2), y = next(data)
        controller = BreakpointController.__init_dict__(model, plugin_cfg)
        controller.cuda()
        # print(controller.breakpoints)
        # for key in Breakpoint.list_of_breakpoints.keys():
        #     for bp in Breakpoint.list_of_breakpoints[key]:
        #         if isinstance(bp.callback, nn.Module):
        #             print(f"{bp.name}: To cuda")
        #             bp.callback.cuda()
        y = model(x1.cuda(), x2.cuda())
        for bp in controller.breakpoints:
            print(bp["breakpoint"].trace)
    
    main()