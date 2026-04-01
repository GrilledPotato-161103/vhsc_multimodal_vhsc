import torch
import torch.nn as nn
import torch.fx as fx
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple


@dataclass
class FxNodeInfo:
    node_name: str
    op: str
    target: Any
    module_path: Optional[str]
    module_type: Optional[str]
    depth: int
    args_repr: str
    kwargs_repr: str


class FxDepthVisualizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.named_modules_map: Dict[str, nn.Module] = dict(model.named_modules())

    def trace(
        self,
        example_inputs: Tuple[Any, ...] | Any,
        tracer_cls: type[fx.Tracer] = fx.Tracer,
    ) -> fx.GraphModule:
        if not isinstance(example_inputs, tuple):
            example_inputs = (example_inputs,)

        tracer = tracer_cls()
        graph = tracer.trace(self.model)
        gm = fx.GraphModule(self.model, graph)
        return gm

    @staticmethod
    def _module_depth(module_path: Optional[str]) -> int:
        if module_path is None or module_path == "":
            return 0
        return module_path.count(".") + 1

    def _resolve_module_info(self, node: fx.Node) -> tuple[Optional[str], Optional[str], int]:
        module_path = None
        module_type = None
        depth = 0

        if node.op == "call_module":
            module_path = str(node.target)
            mod = self.named_modules_map.get(module_path, None)
            module_type = mod.__class__.__name__ if mod is not None else None
            depth = self._module_depth(module_path)

        elif node.op in ("call_function", "call_method"):
            # No direct module owner in FX for these nodes.
            module_path = None
            module_type = None
            depth = 0

        elif node.op in ("placeholder", "output", "get_attr"):
            depth = 0

        return module_path, module_type, depth

    def extract_node_infos(self, gm: fx.GraphModule) -> list[FxNodeInfo]:
        infos = []
        for node in gm.graph.nodes:
            module_path, module_type, depth = self._resolve_module_info(node)
            infos.append(
                FxNodeInfo(
                    node_name=node.name,
                    op=node.op,
                    target=node.target,
                    module_path=module_path,
                    module_type=module_type,
                    depth=depth,
                    args_repr=repr(node.args),
                    kwargs_repr=repr(node.kwargs),
                )
            )
        return infos

    def print_graph_with_depth(
        self,
        gm: fx.GraphModule,
        show_args: bool = False,
        indent_unit: str = "  ",
    ) -> None:
        infos = self.extract_node_infos(gm)

        print("=" * 120)
        print("FX GRAPH WITH MODULE DEPTH")
        print("=" * 120)

        for info in infos:
            indent = indent_unit * info.depth

            module_desc = ""
            if info.module_path is not None:
                module_desc = f" [{info.module_path}: {info.module_type}]"

            line = (
                f"{indent}- {info.node_name:<20} "
                f"op={info.op:<12} "
                f"target={str(info.target):<30} "
                f"depth={info.depth}{module_desc}"
            )
            print(line)

            if show_args:
                print(f"{indent}    args   = {info.args_repr}")
                print(f"{indent}    kwargs = {info.kwargs_repr}")

        print("=" * 120)

    def to_text_tree(self, gm: fx.GraphModule, indent_unit: str = "  ") -> str:
        infos = self.extract_node_infos(gm)
        lines = []

        for info in infos:
            indent = indent_unit * info.depth
            module_desc = ""
            if info.module_path is not None:
                module_desc = f" [{info.module_path}: {info.module_type}]"

            lines.append(
                f"{indent}- {info.node_name} | op={info.op} | target={info.target} | depth={info.depth}{module_desc}"
            )

        return "\n".join(lines)


# -------------------------
# Example usage
# -------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final linear layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        batch_size = x.size(0)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        # Forward through RNN
        out, hn = self.rnn(x, h0)
        # out shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)

        # Take the last time step output
        last_out = out[:, -1, :]
        # shape: (batch_size, hidden_size)

        # Pass through fully connected layer
        y = self.fc(last_out)
        # shape: (batch_size, output_size)

        return y


# Example usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    input_size = 3
    hidden_size = 4
    output_size = 2

    model = SimpleRNN(input_size, hidden_size, output_size)

    # Example input
    x = torch.randn(batch_size, seq_len, input_size)

    vis = FxDepthVisualizer(model)
    
    gm = vis.trace(x)

    print("\nRaw FX graph:")
    print(gm.graph)

    print("\nAnnotated graph:")
    vis.print_graph_with_depth(gm, show_args=False)

    print("\nAs text tree:")
    print(vis.to_text_tree(gm))