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

class Breakpoint(nn.Module):
    list_of_breakpoints: Dict[str, Breakpoint] = defaultdict(list)

    def __init__(
        self,
        name: str,
        callback: Optional[Callable[[BreakpointContext], Any]] = None,
        mutate: bool = False,
        valid: bool = False,
        kwargs: dict  = dict(),
    ):
        super().__init__()
        self.callback = callback
        self.mutate = mutate
        self.valid = valid
        self.trace = None
        self.kwargs = kwargs
        Breakpoint.list_of_breakpoints[name].append(self)
        self.name = f"{name}.{len(Breakpoint.list_of_breakpoints[name]) - 1}"
    
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
        if self.callback is None:
            return inputs, kwargs

        ctx = BreakpointContext(
            name=self.name,
            layer=layer_name,
            position="before",
            module=module,
            inputs=inputs,
            kwargs=kwargs,
            bp_kwargs=self.kwargs,
            state=state,
        )
        result = self.callback(ctx)
        self.trace = result
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
        if self.callback is None:
            return output

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
        )
        result = self.callback(ctx)
        self.trace = result
        if self.mutate and result.output is not None:
            return result.output
        return output


class BreakpointController:

    def __init__(self):
        self.breakpoints: List[Dict[str, Any]] = []
        self.handles: List[Any] = []
        self.state: Dict[str, Any] = {}

    @staticmethod
    def __init_dict__(model: nn.Module, cfg: DictConfig) -> BreakpointController:
        controller = BreakpointController()
        assert type(model).__name__ == cfg.target, "Plugins are going to be plugged into wrong model."
        for item in cfg.breakpoints: 
            bp = instantiate(item.bp)
            controller.add_breakpoint_by_name(model, item.layer_name, bp, item.pos)
        return controller

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
        return [(breakpoint["breakpoint"], (breakpoint["breakpoint"].output, breakpoint["breakpoint"].valid)) for breakpoint in self.breakpoints]

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
        return handle

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
            }
            for item in self.breakpoints
        ]

    def list_breakpoints(self) -> List[Dict[str, Any]]:
        return [
            {
                "layer_name": item["layer_name"],
                "position": item["position"],
                "breakpoint_name": item["breakpoint"].name,
                "module_type": type(item["module"]).__name__,
                "mutate": item["breakpoint"].mutate,
                "callback_key": item["breakpoint"].callback_key,
            }
            for item in self.breakpoints
        ]

    def export_config(self) -> Dict[str, Any]:
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
                    "callback_key": item["breakpoint"].callback_key,
                }
                for item in self.breakpoints
            ]
        }

    def save(self, path: str, use_torch: bool = True):
        data = self.export_config()
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
            raise error if a layer name or callback_key cannot be resolved.
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
            callback_key = spec.get("callback_key")
            callback = None

            if callback_key is not None:
                callback = self.callback_registry.get(callback_key)
                if callback is None:
                    msg = f"Callback key '{callback_key}' is not registered."
                    if strict:
                        raise ValueError(msg)
                    skipped.append({"spec": spec, "reason": msg})
                    continue

            bp = Breakpoint(
                name=spec["name"],
                callback=callback,
                mutate=spec.get("mutate", False),
                callback_key=callback_key,
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

        return {
            "loaded": loaded,
            "skipped": skipped,
        }

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.breakpoints.clear()

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