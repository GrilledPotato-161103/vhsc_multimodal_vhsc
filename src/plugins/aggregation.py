"""
.. deprecated::
    The DAG functionality has been merged into :mod:`src.plugins.hook_dag`.
    Use :class:`~src.plugins.hook_dag.Breakpoint` with ``data_sources``
    instead of :class:`AggregationSpec` / :class:`HookDAG`.

    See :doc:`docs/HOOK_MODULES` for the new DAG-in-breakpoint pattern.

Aggregation Hook System with DAG Topology (DEPRECATED)
=======================================================

Provides a hook structure that aggregates many layers' outputs to generate
new input for its hook layer. The aggregation topology is strictly a
Directed Acyclic Graph (DAG) — nodes that collect from upstream sources
and feed into downstream targets, with cycle detection enforced at
construction time.

Async Subscriber Pattern
------------------------
Each input source is backed by an :class:`Endpoint` in source mode — an
independent "async" subscriber that receives data whenever its corresponding
breakpoint fires during the forward pass. Sources fire at different times
(different layers in the model), and each endpoint buffers its data
independently.  When a node's readiness condition is met (all endpoints have
delivered data, or a threshold is crossed), the aggregation function fires
and the result is dispatched to the target layer.

Because PyTorch forward passes are synchronous, "async" here refers to the
**decoupled, event-driven** nature of the endpoints: each one is a
standalone receiver that does not block or coordinate with other endpoints.
The aggregation node acts as the synchronisation point.

DAG Chaining
------------
Aggregation nodes can be chained: the output of node A can serve as a source
for node B. In this case:

- B subscribes to A by referencing ``@aggregation.<node_a_name>`` in its
  source specification.
- When A fires, it pushes its aggregated output to B's source endpoint.
- The DAG validator ensures no cycles exist.

Integration with BreakpointController
--------------------------------------
:class:`AggregationController` registers the necessary breakpoints with an
existing :class:`BreakpointController`:

- **Source breakpoints** are non-mutating observers that feed endpoints.
- **Target breakpoints** are mutating interceptors that inject the aggregated
  output into the target layer (replacing inputs for ``before`` position or
  outputs for ``after`` position).

Usage Example
-------------

.. code-block:: python

    from src.plugins.aggregation import (
        AggregationController, AggregationSpec, SourceSpec, TargetSpec,
    )
    from src.plugins.hook import BreakpointController

    specs = [
        AggregationSpec(
            name="fuse_features",
            sources=[
                SourceSpec(layer="encoder.layer1", position="after", key="feat_low"),
                SourceSpec(layer="encoder.layer3", position="after", key="feat_high"),
            ],
            target=TargetSpec(layer="decoder.layer1", position="before"),
            aggregate_fn=lambda collected: torch.cat(
                [collected["feat_low"], F.interpolate(collected["feat_high"],
                 size=collected["feat_low"].shape[-2:])], dim=1
            ),
        ),
    ]

    ctrl = BreakpointController()
    agg_ctrl = AggregationController(specs)
    agg_ctrl.register(ctrl, model)
    output = model(x)
    agg_ctrl.reset()  # reset endpoints between passes
"""

from __future__ import annotations

import rootutils
rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import (
    Dict, Optional, Any, Callable, Union, List, Tuple, Set, Sequence,
)

from src.plugins.var import BreakpointContext, BreakpointOutput
from src.plugins.hook import Breakpoint, BreakpointController

# ---------------------------------------------------------------------------
# Sentinel for node-to-node source references
# ---------------------------------------------------------------------------

NODE_SOURCE_PREFIX: str = "@node."


def is_node_source(key: str) -> bool:
    """Return True if *key* references another aggregation node's output."""
    return key.startswith(NODE_SOURCE_PREFIX)


def node_source_name(key: str) -> str:
    """Extract the aggregation node name from a node-source key."""
    return key[len(NODE_SOURCE_PREFIX):]


# ---------------------------------------------------------------------------
# Specification dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    """Describes one upstream source whose output feeds into an aggregation node.

    A source can be:

    - **Layer source**: ``layer`` names a module in ``model.named_modules()``.
      The breakpoint at that layer delivers data whenever the layer fires.
    - **Node source**: ``layer`` starts with ``"@node."`` and the suffix is
      the name of another :class:`AggregationSpec`.  The subscriber receives
      the upstream node's aggregated output when it fires.

    Parameters
    ----------
    layer:
        Layer name (e.g. ``"encoder.0.conv"``) or node reference
        (``"@node.<name>"``).
    position:
        ``"before"`` or ``"after"`` — when to capture data relative to the
        layer's forward call.  Ignored for node sources.
    key:
        Key under which the captured data is stored in the collected dict
        passed to ``aggregate_fn``.
    transform:
        Optional callable applied to the captured data before buffering.
    """
    layer: str
    position: str = "after"
    key: str = ""
    transform: Optional[Callable[[Any], Any]] = None

    def __post_init__(self):
        if not self.key:
            # Default key = layer name with dots replaced by underscores
            self.key = self.layer.replace(".", "_").replace("@", "")
        if self.position not in ("before", "after"):
            raise ValueError(
                f"SourceSpec.position must be 'before' or 'after', got '{self.position}'"
            )


@dataclass
class TargetSpec:
    """Describes where the aggregated output is injected.

    Parameters
    ----------
    layer:
        Target layer name in ``model.named_modules()``.
    position:
        ``"before"`` → inject aggregated output into the layer's *inputs*
        (replaces the first positional argument, or a specific kwarg if
        ``input_key`` is set).

        ``"after"`` → replace the layer's *output* entirely.
    input_key:
        When ``position == "before"``, the name of the keyword argument to
        populate with the aggregated output.  If ``None``, the aggregated
        output replaces the first positional input.
    """
    layer: str
    position: str = "before"
    input_key: Optional[str] = None

    def __post_init__(self):
        if self.position not in ("before", "after"):
            raise ValueError(
                f"TargetSpec.position must be 'before' or 'after', got '{self.position}'"
            )


@dataclass
class AggregationSpec:
    """Complete specification of one aggregation node in the DAG.

    Parameters
    ----------
    name:
        Unique node name.  Other nodes can reference this node's output
        via ``SourceSpec(layer="@node.<name>", ...)``.
    sources:
        Upstream sources whose outputs are collected.
    target:
        Where the aggregated result is injected.
    aggregate_fn:
        Callable ``(Dict[str, Any]) -> Any`` that combines the collected
        source outputs.  Receives a dict keyed by each source's ``key``.
    mode:
        Firing mode:

        - ``"all"`` (default): wait until every subscriber has delivered data.
        - ``"any"``: fire on *every* new source arrival (the aggregate_fn
          always sees all data collected so far; use ``len(collected)`` to
          handle partial inputs gracefully).
    min_sources:
        Minimum number of sources that must have fired before aggregation
        triggers.  Only meaningful when ``mode == "all"`` and a subset of
        sources is sufficient.  Defaults to ``len(sources)``.
    post_transform:
        Optional callable applied to ``aggregate_fn``'s output before
        injection.  Receives the aggregated value, returns the transformed
        value.
    """
    name: str
    sources: List[SourceSpec] = field(default_factory=list)
    target: TargetSpec = field(default_factory=TargetSpec)
    aggregate_fn: Optional[Callable[[Dict[str, Any]], Any]] = None
    mode: str = "all"
    min_sources: Optional[int] = None
    post_transform: Optional[Callable[[Any], Any]] = None

    def __post_init__(self):
        if self.mode not in ("all", "any"):
            raise ValueError(
                f"AggregationSpec.mode must be 'all' or 'any', got '{self.mode}'"
            )
        if self.min_sources is None:
            self.min_sources = len(self.sources)
        if self.min_sources < 1:
            raise ValueError(f"min_sources must be >= 1, got {self.min_sources}")
        if self.min_sources > len(self.sources):
            raise ValueError(
                f"min_sources ({self.min_sources}) cannot exceed "
                f"number of sources ({len(self.sources)})"
            )

    def default_aggregate_fn(self) -> Callable[[Dict[str, Any]], Any]:
        """Return a sensible fallback when no ``aggregate_fn`` is provided.

        If all collected values are tensors with matching shapes, concatenates
        along the last dimension.  Otherwise returns the collected dict as-is.
        """
        def _fallback(collected: Dict[str, Any]) -> Any:
            tensors = []
            for v in collected.values():
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
                else:
                    # Non-tensor value — return collected dict unchanged
                    return collected
            if not tensors:
                return collected
            if len(tensors) == 1:
                return tensors[0]
            # Attempt concatenation; if shapes differ, stack along a new dim
            try:
                return torch.cat(tensors, dim=-1)
            except RuntimeError:
                return torch.stack(tensors, dim=0)
        return _fallback


# ---------------------------------------------------------------------------
# Unified Endpoint — breakpoint callback for both source and target layers
# ---------------------------------------------------------------------------

class Endpoint(nn.Module):
    """Unified breakpoint callback for aggregation sources and targets.

    Replaces the separate ``InputSubscriber`` (source observer) and
    ``AggregationNode.forward()`` (target injector) with a single class
    parameterized by the direction of data flow:

    **Source mode** (``on_receive`` is set):
        Captures data from :class:`BreakpointContext`, applies an optional
        transform, and pushes it to the parent :class:`AggregationNode`
        via the ``on_receive`` callback.  Non-mutating — the layer's output
        passes through unchanged.

    **Target mode** (``coordinator`` is set):
        Reads the aggregated output from the parent
        :class:`AggregationNode`.  If the node has fired, injects the
        aggregated value into the target layer (replacing inputs for
        ``"before"`` position or the output for ``"after"``).  Mutating.

    The two modes are mutually exclusive.

    Parameters
    ----------
    key:
        Identifier for this endpoint (used as the key in the collected dict
        for source mode, or as the node name for target mode).
    transform:
        Optional callable applied to captured data before buffering
        (source mode only).
    on_receive:
        Callback ``(key, data)`` invoked when data is captured (source mode).
        Typically :meth:`AggregationNode._on_source_data`.
    coordinator:
        The parent :class:`AggregationNode` whose ``ready`` / ``aggregated``
        state is read at injection time (target mode).
    position:
        ``"before"`` or ``"after"`` — injection point for target mode.
    input_key:
        When target mode and ``position == "before"``, the kwarg name to
        populate with the aggregated output.  If ``None``, the aggregated
        value replaces the first positional input.
    """

    def __init__(
        self,
        *,
        key: str = "",
        transform: Optional[Callable[[Any], Any]] = None,
        on_receive: Optional[Callable[[str, Any], None]] = None,
        coordinator: Optional[nn.Module] = None,
        position: str = "after",
        input_key: Optional[str] = None,
    ):
        super().__init__()
        if on_receive is not None and coordinator is not None:
            raise ValueError(
                "Endpoint cannot be both source (on_receive) and target (coordinator)"
            )
        if on_receive is None and coordinator is None:
            raise ValueError(
                "Endpoint must be either source (on_receive) or target (coordinator)"
            )
        self._key = key
        self._transform = transform
        self._on_receive = on_receive
        # Use object.__setattr__ to prevent PyTorch from auto-registering
        # the coordinator as a sub-module (which would create a cycle:
        # AggregationNode → target_endpoint → coordinator → AggregationNode).
        object.__setattr__(self, '_coordinator', coordinator)
        self._position = position
        self._input_key = input_key
        # Source-mode buffer
        self._data: Any = None
        self._arrived: bool = False

    # --- breakpoint callback contract ---------------------------------------

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        """Breakpoint callback — dispatch on source vs target mode."""
        if self._on_receive is not None:
            return self._forward_source(ctx)
        return self._forward_target(ctx)

    # --- source mode --------------------------------------------------------

    def _forward_source(self, ctx: BreakpointContext) -> BreakpointOutput:
        """Capture data from *ctx*, transform, and push to the coordinator."""
        data = ctx.output if ctx.position == "after" else ctx.inputs
        if self._transform is not None:
            data = self._transform(data)
        self._data = data
        self._arrived = True
        if self._on_receive is not None:
            self._on_receive(self._key, data)
        return BreakpointOutput(
            fn_name=f"endpoint.source.{self._key}",
            context=ctx,
            output=data,  # pass-through, non-mutating
            trace={"source_key": self._key},
            valid=True,
        )

    # --- target mode --------------------------------------------------------

    def _forward_target(self, ctx: BreakpointContext) -> BreakpointOutput:
        """Inject aggregated output if the coordinator has fired."""
        coord = self._coordinator
        if coord is None or not coord.ready:
            origin = ctx.output if self._position == "after" else ctx.inputs
            return BreakpointOutput(
                fn_name=f"endpoint.target.{self._key}",
                context=ctx,
                output=origin,
                trace={"warning": f"Node '{self._key}' not ready"},
                valid=False,
            )

        aggregated = coord.aggregated

        if self._position == "before":
            if self._input_key is not None:
                # Mutate kwargs in-place — ctx.kwargs is the same dict that
                # run_before returns to the PyTorch hook, so the mutation
                # propagates to the module call.
                ctx.kwargs[self._input_key] = aggregated
                return BreakpointOutput(
                    fn_name=f"endpoint.target.{self._key}",
                    context=ctx,
                    output=ctx.inputs,  # keep original positional args
                    trace={
                        "aggregated": aggregated,
                        "input_key": self._input_key,
                    },
                    valid=True,
                )
            else:
                new_inputs = (aggregated,) + ctx.inputs[1:]
                return BreakpointOutput(
                    fn_name=f"endpoint.target.{self._key}",
                    context=ctx,
                    output=new_inputs,
                    trace={"aggregated": aggregated},
                    valid=True,
                )
        else:
            return BreakpointOutput(
                fn_name=f"endpoint.target.{self._key}",
                context=ctx,
                output=aggregated,
                trace={"aggregated": aggregated, "original_output": ctx.output},
                valid=True,
            )

    # --- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear buffered data (call between forward passes)."""
        self._data = None
        self._arrived = False

    # --- properties ---------------------------------------------------------

    @property
    def data(self) -> Any:
        """The most recently received data (or ``None``) — source mode."""
        return self._data

    @property
    def ready(self) -> bool:
        """``True`` if data has been received since the last :meth:`reset` — source mode."""
        return self._arrived

    @property
    def is_source(self) -> bool:
        """``True`` if this endpoint is in source (observer) mode."""
        return self._on_receive is not None

    @property
    def is_target(self) -> bool:
        """``True`` if this endpoint is in target (injector) mode."""
        return self._coordinator is not None


# ---------------------------------------------------------------------------
# Aggregation Node — one vertex in the DAG (pure coordinator)
# ---------------------------------------------------------------------------

class AggregationNode(nn.Module):
    """One vertex in the aggregation DAG — a pure coordinator.

    An :class:`AggregationNode` no longer serves as a breakpoint callback
    itself.  Instead it owns:

    - **Source endpoints** (:class:`Endpoint` in source mode) — one per
      upstream source.  Each is a breakpoint callback that captures data
      and pushes it to this node.
    - **One target endpoint** (:class:`Endpoint` in target mode) — the
      breakpoint callback that injects the aggregated output at the target
      layer.

    When enough sources have delivered data the node:

    1. Calls ``spec.aggregate_fn(collected_dict)``.
    2. Optionally applies ``spec.post_transform``.
    3. Stores the result in :attr:`aggregated`.
    4. Notifies any downstream nodes that subscribe to this node's output.

    Parameters
    ----------
    spec:
        The :class:`AggregationSpec` describing sources, target, and
        aggregation function.
    """

    def __init__(self, spec: AggregationSpec):
        super().__init__()
        self.spec = spec
        self._collected: Dict[str, Any] = {}
        self._aggregated: Any = None
        self._fired: bool = False
        # Downstream nodes that subscribe to this node's output.
        self._downstream: Dict[AggregationNode, str] = {}

        # Source endpoints — each is a breakpoint callback (non-mutating).
        ep_dict: Dict[str, Endpoint] = {}
        for src in spec.sources:
            ep = Endpoint(
                key=src.key,
                transform=src.transform,
                on_receive=self._on_source_data,
            )
            ep_dict[src.key] = ep
        self.source_endpoints = nn.ModuleDict(ep_dict)

        # Target endpoint — breakpoint callback (mutating).
        self.target_endpoint = Endpoint(
            key=spec.name,
            coordinator=self,
            position=spec.target.position,
            input_key=spec.target.input_key,
        )

    # --- internal -----------------------------------------------------------

    def _on_source_data(self, key: str, data: Any) -> None:
        """Callback invoked by a source endpoint when new data arrives."""
        self._collected[key] = data
        if self._should_fire():
            self._fire()

    def _should_fire(self) -> bool:
        """Return ``True`` if the readiness condition is satisfied."""
        n = len(self._collected)
        if n == 0:
            return False
        if self.spec.mode == "any":
            return True
        return n >= (self.spec.min_sources or len(self.spec.sources))

    def _fire(self) -> None:
        """Execute the aggregation function and notify downstream nodes."""
        fn = self.spec.aggregate_fn or self.spec.default_aggregate_fn()
        result = fn(dict(self._collected))
        if self.spec.post_transform is not None:
            result = self.spec.post_transform(result)
        self._aggregated = result
        self._fired = True

        for downstream, local_key in self._downstream.items():
            downstream._on_source_data(local_key, result)

    def _add_downstream(self, node: AggregationNode, local_key: str) -> None:
        """Register a downstream node that depends on this node's output."""
        if node not in self._downstream:
            self._downstream[node] = local_key

    # --- properties ---------------------------------------------------------

    @property
    def aggregated(self) -> Any:
        """The result of the most recent :meth:`_fire` (or ``None``)."""
        return self._aggregated

    @property
    def ready(self) -> bool:
        """``True`` if the node has fired at least once since :meth:`reset`."""
        return self._fired

    @property
    def collected(self) -> Dict[str, Any]:
        """Read-only view of currently collected source data."""
        return dict(self._collected)

    # --- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Reset all endpoints and internal state (call between passes)."""
        self._collected.clear()
        self._aggregated = None
        self._fired = False
        for ep in self.source_endpoints.values():
            ep.reset()
        self.target_endpoint.reset()


# ---------------------------------------------------------------------------
# DAG validation & topological ordering
# ---------------------------------------------------------------------------

class HookDAG:
    """Validates the DAG topology of a set of aggregation specifications.

    Ensures:

    - **No cycles**: detection via depth-first search with tricolour marking.
    - **No duplicate names**: every :class:`AggregationSpec` must have a
      unique ``name``.
    - **Referential integrity**: every node-source reference
      (``"@node.<name>"``) must point to an existing node.

    After validation the topological execution order is computed (Kahn's
    algorithm) so that upstream nodes always execute before downstream nodes.

    Parameters
    ----------
    specs:
        The aggregation specifications to validate and manage.
    """

    def __init__(self, specs: Sequence[AggregationSpec]):
        self.specs = list(specs)
        self.nodes: Dict[str, AggregationNode] = {}
        self._execution_order: List[str] = []
        self._adj: Dict[str, Set[str]] = defaultdict(set)   # u -> {v: u feeds into v}
        self._rev_adj: Dict[str, Set[str]] = defaultdict(set)  # v -> {u}

        # --- duplicate check ---
        seen: Set[str] = set()
        for spec in self.specs:
            if spec.name in seen:
                raise ValueError(f"Duplicate aggregation node name: '{spec.name}'")
            seen.add(spec.name)

        # --- referential integrity ---
        node_names = {s.name for s in self.specs}
        for spec in self.specs:
            for src in spec.sources:
                if is_node_source(src.layer):
                    ref = node_source_name(src.layer)
                    if ref not in node_names:
                        raise ValueError(
                            f"Node '{spec.name}' references unknown node "
                            f"'{ref}' in source '{src.layer}'"
                        )

        # --- build nodes ---
        for spec in self.specs:
            self.nodes[spec.name] = AggregationNode(spec)

        # --- build adjacency (node-to-node edges) ---
        for spec in self.specs:
            for src in spec.sources:
                if is_node_source(src.layer):
                    upstream = node_source_name(src.layer)
                    self._adj[upstream].add(spec.name)
                    self._rev_adj[spec.name].add(upstream)

        # --- wire downstream pointers (with correct key mapping) ---
        for spec in self.specs:
            dn_node = self.nodes.get(spec.name)
            if dn_node is None:
                continue
            for src in spec.sources:
                if is_node_source(src.layer):
                    upstream = node_source_name(src.layer)
                    up_node = self.nodes.get(upstream)
                    if up_node is not None:
                        up_node._add_downstream(dn_node, src.key)

        # --- cycle detection + topological sort ---
        self._validate_no_cycles()
        self._topological_sort()

    # ------------------------------------------------------------------
    # cycle detection — DFS tricolour
    # ------------------------------------------------------------------

    def _validate_no_cycles(self) -> None:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {s.name: WHITE for s in self.specs}

        def dfs(u: str) -> None:
            color[u] = GRAY
            for v in self._adj.get(u, set()):
                c = color.get(v, BLACK)
                if c == GRAY:
                    raise ValueError(
                        f"Cycle detected in aggregation DAG: "
                        f"'{u}' -> '{v}' (back edge)"
                    )
                if c == WHITE:
                    dfs(v)
            color[u] = BLACK

        for name in color:
            if color[name] == WHITE:
                dfs(name)

    # ------------------------------------------------------------------
    # topological sort — Kahn's algorithm
    # ------------------------------------------------------------------

    def _topological_sort(self) -> None:
        in_degree: Dict[str, int] = {s.name: 0 for s in self.specs}
        for upstream, downstreams in self._adj.items():
            for _ in downstreams:
                in_degree[_] = in_degree.get(_, 0) + 1

        queue: deque = deque(
            [name for name, deg in in_degree.items() if deg == 0]
        )
        order: List[str] = []

        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self._adj.get(u, set()):
                if v in in_degree:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(order) != len(self.specs):
            # Should not happen if cycle detection passed, but guard anyway
            missing = set(self.nodes.keys()) - set(order)
            raise ValueError(
                f"Failed to produce topological order.  "
                f"Unreachable nodes: {missing}.  This indicates a cycle."
            )

        self._execution_order = order

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Export serializable DAG topology (specs without callables).

        Callables (``aggregate_fn``, ``post_transform``, ``transform``) are
        not serialized — they must be re-supplied via the original
        ``AggregationSpec`` list on load.

        Returns
        -------
        dict
            With keys ``"specs"`` (list of serialized spec dicts) and
            ``"nodes"`` (per-node ``nn.Module.state_dict()``).
        """
        return {
            "specs": [
                {
                    "name": s.name,
                    "sources": [
                        {
                            "layer": src.layer,
                            "position": src.position,
                            "key": src.key,
                        }
                        for src in s.sources
                    ],
                    "target": {
                        "layer": s.target.layer,
                        "position": s.target.position,
                        "input_key": s.target.input_key,
                    },
                    "mode": s.mode,
                    "min_sources": s.min_sources,
                }
                for s in self.specs
            ],
            "nodes": {
                name: node.state_dict()
                for name, node in self.nodes.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore node parameters from a checkpoint.

        Only restores ``"nodes"`` — the ``"specs"`` section is used for
        validation when the checkpoint is loaded via
        :meth:`AggregationController.load_checkpoint`.

        Parameters
        ----------
        state_dict:
            The dict previously returned by :meth:`state_dict`.
        """
        node_states = state_dict.get("nodes", {})
        for name, node_state in node_states.items():
            if name in self.nodes:
                self.nodes[name].load_state_dict(node_state, strict=False)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def get_node(self, name: str) -> AggregationNode:
        """Return the :class:`AggregationNode` with the given *name*.

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        if name not in self.nodes:
            raise KeyError(
                f"Aggregation node '{name}' not found. "
                f"Available: {list(self.nodes.keys())}"
            )
        return self.nodes[name]

    @property
    def execution_order(self) -> List[str]:
        """Topological order of node names (upstream → downstream)."""
        return list(self._execution_order)

    def reset_all(self) -> None:
        """Reset every node (subscribers + internal state)."""
        for node in self.nodes.values():
            node.reset()

    def summary(self) -> str:
        """Return a human-readable summary of the DAG topology."""
        lines = [f"HookDAG ({len(self.nodes)} nodes):"]
        for name in self._execution_order:
            node = self.nodes[name]
            src_keys = list(node.source_endpoints.keys())
            deps = sorted(self._rev_adj.get(name, set()))
            feeds = sorted(self._adj.get(name, set()))
            lines.append(
                f"  [{name}]"
                f"  sources={src_keys}"
                + (f"  depends_on={deps}" if deps else "")
                + (f"  feeds={feeds}" if feeds else "")
                + f"  target={node.spec.target.layer}:{node.spec.target.position}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation Controller — integrates with BreakpointController
# ---------------------------------------------------------------------------

class AggregationController:
    """Orchestrates the registration of aggregation breakpoints onto a model.

    Given a list of :class:`AggregationSpec` objects and an existing
    :class:`BreakpointController`, :meth:`register` creates:

    - **Source breakpoints**: each :class:`Endpoint` (source mode) **is** the
      callback — its :meth:`~Endpoint.forward` method captures data and feeds
      the parent :class:`AggregationNode`.  Non-mutating.
    - **Target breakpoint**: the :class:`Endpoint` (target mode) **is** the
      callback — its :meth:`~Endpoint.forward` method injects the aggregated
      output into the target layer.  Mutating.

    Node-to-node sources (``"@node.<name>"``) do **not** get a layer
    breakpoint — they are fed directly when the upstream node fires.

    Parameters
    ----------
    specs:
        One or more :class:`AggregationSpec` definitions.
    """

    def __init__(self, specs: Sequence[AggregationSpec]):
        self.dag = HookDAG(specs)
        # Track created breakpoints so they can be inspected / cleared
        self._source_breakpoints: Dict[str, List[Breakpoint]] = defaultdict(list)
        self._target_breakpoints: Dict[str, Breakpoint] = {}

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(
        self,
        controller: BreakpointController,
        model: nn.Module,
    ) -> AggregationController:
        """Register all aggregation source and target breakpoints.

        Source breakpoints use :class:`Endpoint` (source mode) as the
        callback — ``forward(ctx)`` captures data and feeds the parent node.
        Target breakpoints use :class:`Endpoint` (target mode) as the
        callback — ``forward(ctx)`` injects the aggregated output.

        Parameters
        ----------
        controller:
            An existing :class:`BreakpointController` that manages hook
            registration on *model*.
        model:
            The PyTorch model to attach breakpoints to.

        Returns
        -------
        self
            For method chaining.
        """
        for spec in self.dag.specs:
            node = self.dag.get_node(spec.name)

            # --- source breakpoints (layer-based only) ---
            # Endpoint.forward(ctx) in source mode IS the subscribe callback.
            for src in spec.sources:
                if is_node_source(src.layer):
                    # Node-to-node sources are wired via _add_downstream
                    continue

                ep = node.source_endpoints[src.key]
                bp = Breakpoint(
                    name=f"agg.{spec.name}.src.{src.key}",
                    callback=ep,                # Endpoint.forward(ctx) — source mode
                    mutate=False,
                )
                controller.add_breakpoint(
                    root=model,
                    target=src.layer,
                    bp=bp,
                    position=src.position,
                )
                self._source_breakpoints[spec.name].append(bp)

            # --- target breakpoint ---
            # Endpoint.forward(ctx) in target mode IS the inject callback.
            bp = Breakpoint(
                name=f"agg.{spec.name}.tgt",
                callback=node.target_endpoint,  # Endpoint.forward(ctx) — target mode
                mutate=True,
            )
            controller.add_breakpoint(
                root=model,
                target=spec.target.layer,
                bp=bp,
                position=spec.target.position,
            )
            self._target_breakpoints[spec.name] = bp

        return self

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all subscribers and nodes (call between forward passes)."""
        self.dag.reset_all()

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, AggregationNode]:
        """Dict of node name → :class:`AggregationNode`."""
        return self.dag.nodes

    def get_aggregated(self, name: str) -> Any:
        """Return the aggregated output of node *name* (or ``None``)."""
        return self.dag.get_node(name).aggregated

    def summary(self) -> str:
        """Return a human-readable topology summary."""
        return self.dag.summary()

    # ------------------------------------------------------------------
    # checkpoint — save / load
    # ------------------------------------------------------------------

    def state_dict(self, controller: BreakpointController) -> Dict[str, Any]:
        """Export full serializable state: HookDAG topology + breakpoints.

        Callbacks (``nn.Module`` instances) are saved via their
        ``state_dict()``, keyed by breakpoint name.  The breakpoint
        configuration (layer name, position, mutate flag) is stored
        separately so hooks can be re-attached on load.

        Callables (``aggregate_fn``, ``post_transform``, ``transform``) are
        **not** serialized — pass the original spec list to
        :meth:`load_checkpoint`.

        Parameters
        ----------
        controller:
            The :class:`BreakpointController` whose breakpoints should be
            saved alongside the DAG state.

        Returns
        -------
        dict
            Keys: ``"hook_dag"``, ``"breakpoints"`` (list of config dicts),
            ``"callback_params"`` (dict of breakpoint name → state_dict).
        """
        # Breakpoint config without actual callback objects (unpicklable)
        bp_config: List[Dict[str, Any]] = []
        callback_params: Dict[str, Any] = {}
        for item in controller.breakpoints:
            bp = item["breakpoint"]
            bp_config.append({
                "name": bp.name,
                "layer_name": item["layer_name"],
                "position": item["position"],
                "mutate": bp.mutate,
            })
            if isinstance(bp.callback, nn.Module):
                callback_params[bp.name] = bp.callback.state_dict()

        return {
            "hook_dag": self.dag.state_dict(),
            "breakpoints": bp_config,
            "callback_params": callback_params,
        }

    def save(
        self,
        controller: BreakpointController,
        path: str,
    ) -> None:
        """Persist HookDAG topology + breakpoint state to *path*.

        Uses :func:`torch.save`.  Callables (lambdas, bound methods) are
        stripped — re-supply the original :class:`AggregationSpec` list on
        load via :meth:`load_checkpoint`.

        Parameters
        ----------
        controller:
            The :class:`BreakpointController` to save alongside the DAG.
        path:
            File path for the checkpoint (e.g. ``"plugins.pt"``).
        """
        torch.save(self.state_dict(controller), path)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback parameters from a checkpoint dict.

        Parameters
        ----------
        state_dict:
            The dict previously returned by :meth:`state_dict`.
        """
        self.dag.load_state_dict(state_dict.get("hook_dag", {}))

    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        specs: Sequence[AggregationSpec],
    ) -> Tuple[BreakpointController, AggregationController]:
        """Load a checkpoint and reconstruct the full hook DAG.

        Reconstructs the :class:`HookDAG` from *specs* (which must include
        the original callables), re-attaches all breakpoint hooks, and
        restores saved callback parameters.

        Parameters
        ----------
        path:
            Path to a checkpoint previously created by :meth:`save`.
        model:
            The PyTorch model to re-attach hooks to.
        specs:
            The original :class:`AggregationSpec` list (with callables).

        Returns
        -------
        Tuple[BreakpointController, AggregationController]
            ``(controller, agg_ctrl)`` with all hooks re-attached and
            parameters restored.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)

        # Step 1 — build the aggregation controller (creates Endpoint callbacks)
        agg_ctrl = AggregationController(specs)

        # Step 2 — create BreakpointController and register aggregation breakpoints
        bp_ctrl = BreakpointController()
        agg_ctrl.register(bp_ctrl, model)

        # Step 3 — restore callback parameters (Endpoint + user callbacks)
        agg_ctrl.load_state_dict(data)

        # Step 4 — load saved params into the registered breakpoint callbacks
        callback_params = data.get("callback_params", {})
        for item in bp_ctrl.breakpoints:
            bp_name = item["breakpoint"].name
            if bp_name in callback_params:
                cb = item["breakpoint"].callback
                if isinstance(cb, nn.Module):
                    cb.load_state_dict(callback_params[bp_name], strict=False)

        return bp_ctrl, agg_ctrl

    @staticmethod
    def from_config(
        model: nn.Module,
        cfg: AggregationSpec | Sequence[AggregationSpec],
    ) -> AggregationController:
        """Build an :class:`AggregationController` directly from spec objects.

        Convenience wrapper for when you are not using Hydra configs.

        Parameters
        ----------
        model:
            The model (not used during construction — registration happens
            later via :meth:`register`).
        cfg:
            A single :class:`AggregationSpec` or a list of them.

        Returns
        -------
        AggregationController
            Ready to call ``.register(controller, model)``.
        """
        if isinstance(cfg, AggregationSpec):
            cfg = [cfg]
        return AggregationController(cfg)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F

    print("=" * 60)
    print("Aggregation DAG — Smoke Test")
    print("=" * 60)

    # --- Build a toy model for testing ---
    class ToyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            f1 = torch.relu(self.conv1(x))       # layer: conv1
            f2 = torch.relu(self.conv2(f1))       # layer: conv2
            return self.pool(f2)

    class ToyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16 + 8, 10)

        def forward(self, fused):
            return self.fc(fused)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ToyEncoder()
            self.decoder = ToyDecoder()

        def forward(self, x):
            f = self.encoder(x)   # (B, 16, 1, 1)
            # The aggregation will fuse encoder.conv1 output + encoder output
            # and inject as 'fused' into decoder.
            return self.decoder(f)

    model = ToyModel()

    # --- Define aggregation: fuse conv1 output and encoder final output ---
    specs = [
        AggregationSpec(
            name="fuse_early_late",
            sources=[
                SourceSpec(layer="encoder.conv1", position="after", key="conv1_out"),
                SourceSpec(layer="encoder", position="after", key="enc_out"),
            ],
            target=TargetSpec(layer="decoder", position="before"),
            aggregate_fn=lambda collected: torch.cat(
                [
                    collected["conv1_out"].mean(dim=[-2, -1]),   # (B, 8)
                    collected["enc_out"].squeeze(-1).squeeze(-1),  # (B, 16)
                ],
                dim=-1,
            ),
        ),
    ]

    # --- Wire everything up ---
    bp_ctrl = BreakpointController()
    agg_ctrl = AggregationController(specs)
    agg_ctrl.register(bp_ctrl, model)

    print(agg_ctrl.summary())
    print()

    # --- Forward pass ---
    x = torch.randn(2, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {y.shape}")         # expected: (2, 10)
    fused = agg_ctrl.get_aggregated("fuse_early_late")
    print(f"Aggregated shape: {fused.shape}")  # expected: (2, 24)

    # Inspect breakpoints
    for bp_info in bp_ctrl.breakpoints:
        t = bp_info["breakpoint"].trace
        if t is not None:
            print(f"  {bp_info['breakpoint'].name}: valid={t.valid}")

    # --- Reset and second pass ---
    agg_ctrl.reset()
    with torch.no_grad():
        y2 = model(x)
    print(f"Second pass output shape: {y2.shape}")
    print()

    # --- DAG chaining test ---
    print("--- DAG Chaining Test ---")

    class ToyChainModel(nn.Module):
        """Model whose dimensions are consistent with the 2-stage chain below."""
        def __init__(self):
            super().__init__()
            self.enc_conv1 = nn.Conv2d(3, 8, 3, padding=1)   # → (B, 8, 32, 32)
            self.enc_relu = nn.ReLU()
            self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)  # → (B, 16, 32, 32)
            self.pool = nn.AdaptiveAvgPool2d(1)               # → (B, 16, 1, 1)
            # stage1: conv1 (8ch) + conv2 (16ch) → cat → (24, 32, 32)
            # stage1 replaces conv2's output, so pool gets (24, 32, 32) → (24, 1, 1)
            # stage2: stage1_out (24ch) mean → (24,) + enc_out squeeze → (24,) → cat → (48,)
            self.decoder = nn.Linear(48, 10)

        def forward(self, x):
            c1 = self.enc_relu(self.enc_conv1(x))
            c2 = self.enc_relu(self.enc_conv2(c1))
            pooled = self.pool(c2)
            return self.decoder(pooled)

    model2 = ToyChainModel()

    specs_chain = [
        AggregationSpec(
            name="stage1",
            sources=[
                SourceSpec(layer="enc_conv1", position="after", key="c1"),
                SourceSpec(layer="enc_conv2", position="after", key="c2"),
            ],
            target=TargetSpec(layer="enc_conv2", position="after"),
            aggregate_fn=lambda d: torch.cat([d["c1"], d["c2"]], dim=1),
        ),
        AggregationSpec(
            name="stage2",
            sources=[
                SourceSpec(layer="@node.stage1", position="after", key="from_stage1"),
                SourceSpec(layer="pool", position="after", key="final"),
            ],
            target=TargetSpec(layer="decoder", position="before"),
            aggregate_fn=lambda d: torch.cat(
                [d["from_stage1"].mean(dim=[-2, -1]), d["final"].squeeze(-1).squeeze(-1)],
                dim=-1,
            ),
        ),
    ]

    bp_ctrl2 = BreakpointController()
    agg_ctrl2 = AggregationController(specs_chain)
    agg_ctrl2.register(bp_ctrl2, model2)
    print(agg_ctrl2.summary())
    print()

    with torch.no_grad():
        y3 = model2(x)
    print(f"Chained output shape: {y3.shape}")
    print(f"stage1 aggregated shape: {agg_ctrl2.get_aggregated('stage1').shape}")
    print(f"stage2 aggregated shape: {agg_ctrl2.get_aggregated('stage2').shape}")

    # --- Cycle detection test ---
    print("\n--- Cycle Detection Test ---")
    try:
        AggregationController([
            AggregationSpec(
                name="a",
                sources=[SourceSpec(layer="@node.b", key="b_out")],
                target=TargetSpec(layer="encoder", position="before"),
            ),
            AggregationSpec(
                name="b",
                sources=[SourceSpec(layer="@node.a", key="a_out")],
                target=TargetSpec(layer="encoder", position="before"),
            ),
        ])
        print("ERROR: should have raised ValueError for cycle")
    except ValueError as e:
        print(f"Correctly caught cycle: {e}")

    # --- Checkpoint save / load test ---
    print("\n--- Checkpoint Save / Load Test ---")

    # Re-use the basic ToyModel + specs from above
    model3 = ToyModel()
    specs3 = [
        AggregationSpec(
            name="fuse_early_late",
            sources=[
                SourceSpec(layer="encoder.conv1", position="after", key="conv1_out"),
                SourceSpec(layer="encoder", position="after", key="enc_out"),
            ],
            target=TargetSpec(layer="decoder", position="before"),
            aggregate_fn=lambda collected: torch.cat(
                [
                    collected["conv1_out"].mean(dim=[-2, -1]),
                    collected["enc_out"].squeeze(-1).squeeze(-1),
                ],
                dim=-1,
            ),
        ),
    ]

    bp_ctrl3 = BreakpointController()
    agg_ctrl3 = AggregationController(specs3)
    agg_ctrl3.register(bp_ctrl3, model3)

    # Run one forward pass to populate trace
    model3.eval()
    with torch.no_grad():
        y_before = model3(x)

    # Save checkpoint
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    ckpt_path = os.path.join(tmpdir, "hook_dag_test.pt")
    agg_ctrl3.save(bp_ctrl3, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}  ({os.path.getsize(ckpt_path)} bytes)")

    # Clear everything
    bp_ctrl3.clear()
    agg_ctrl3.reset()

    # Load checkpoint (re-supply specs with callables)
    bp_loaded, agg_loaded = AggregationController.load_checkpoint(
        ckpt_path, model3, specs3,
    )
    print(f"Loaded: {agg_loaded.summary().split(chr(10))[0]}")

    # Verify forward pass still works
    with torch.no_grad():
        y_after = model3(x)
    print(f"Output after restore: {y_after.shape}")
    assert torch.allclose(y_before, y_after, atol=1e-5), \
        "Output mismatch after checkpoint round-trip!"
    print("Checkpoint round-trip: output matches OK")

    # Verify aggregated output is correct
    fused_after = agg_loaded.get_aggregated("fuse_early_late")
    print(f"Aggregated shape after restore: {fused_after.shape}")
    assert fused_after is not None, "Aggregation did not fire after restore!"
    print("Checkpoint: aggregation fires correctly OK")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("All smoke tests passed.")
