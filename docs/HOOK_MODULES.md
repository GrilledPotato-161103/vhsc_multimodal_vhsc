# Hook Modules — Architecture & Reference

> Documentation for the breakpoint/hook system, aggregation DAG, reconstructors, uncertainty heads, and propagation pipelines.
> Generated: 2026-06-20

---

## Overview

The hook system is a custom PyTorch forward-hook framework that enables **non-invasive** injection of computation into frozen pretrained models. Instead of modifying the model's `forward()` method, callbacks are registered on specific layers via `register_forward_hook` / `register_forward_pre_hook` and execute at runtime during the forward pass.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BreakpointController                         │
│  Manages registration of Breakpoint instances on model layers       │
│  Provides shared state dict, save/load, lifecycle (eval/train/to)   │
└──────────┬────────────────────────────────────────────┬─────────────┘
           │                                            │
    ┌──────▼──────┐                            ┌───────▼───────┐
    │  Breakpoint  │                            │  Breakpoint    │
    │  name="..."  │                            │  name="..."    │
    │  callback=███│                            │  callback=███  │
    │  mutate=False│                            │  mutate=True   │
    └──────┬──────┘                            └───────┬───────┘
           │                                            │
    ┌──────▼────────────────────────────────────────────▼──────────┐
    │                   Callback Contract                           │
    │  Input:  BreakpointContext(name, layer, position, module,    │
    │           inputs, kwargs, output, bp_kwargs, state)           │
    │  Output: BreakpointOutput(fn_name, context, output, trace,   │
    │           valid)                                              │
    └──────────────────────────────────────────────────────────────┘
```

Two positions per layer:

| Position | Hook type | PyTorch mechanism | Default behavior |
|----------|-----------|-------------------|------------------|
| `"before"` | Pre-hook | `register_forward_pre_hook` | Receives `(inputs, kwargs)`, can replace positional args or mutate kwargs in-place |
| `"after"` | Forward hook | `register_forward_hook` | Receives `(inputs, kwargs, output)`, can replace the module's output |

---

## Module Map

```
src/plugins/
├── hook.py                 # Breakpoint, BreakpointController (original)
├── hook_dag.py             # Breakpoint, BreakpointController (DAG-aware)
├── var.py                  # Data structures + EndpointSpec
├── aggregation.py          # DEPRECATED — use hook_dag.py
├── aggregate.py            # Router / EndpointWrapper (request-based)
├── ekf_propagation.py      # EKF diagonal + full covariance propagation
├── sigma_z.py              # SDSigmaZ, GroundTruthSigmaZ, BNShiftSigmaZ
├── head/
│   ├── ekf.py              # EKFBiModalInferer (GGD head, EKF alpha)
│   ├── bayescap.py         # BayesCap1D + BayesCap1DLoss + EKFGGDNLLLoss
│   └── hessian.py          # HessianBiModalInferer + DiscreteHessianBiModalInferer
└── reconstructor/
    ├── linear.py           # BilinearReconstructor, LinearReconstructor
    └── identity.py         # IdentityHook (no-op passthrough)
```

---

## 1. Core Hook Infrastructure

### `hook.py` — Breakpoint & BreakpointController

#### `Breakpoint(nn.Module)` — Single Hook Instance

Wraps a callback and registers itself into a global registry.

```python
bp = Breakpoint(
    name="my_bp",          # logical name; auto-suffixed with .N index
    callback=my_module,     # Callable[[BreakpointContext], Any]
    mutate=False,           # True → callback can replace layer input/output
    valid=False,            # user-defined validity flag
    kwargs={},              # user kwargs exposed as ctx.bp_kwargs
)
```

**Key methods:**
- `run_before(layer_name, module, inputs, kwargs, state)` — pre-hook handler. Constructs `BreakpointContext(position="before")`, calls `callback(ctx)`, and if `mutate=True` and `result.output` is a `tuple`, returns it as replacement positional args.
- `run_after(layer_name, module, inputs, kwargs, output, state)` — post-hook handler. Constructs `BreakpointContext(position="after")`, calls `callback(ctx)`, and if `mutate=True` and `result.output is not None`, returns it as replacement output.
- `self.trace` — stores the `BreakpointOutput` returned by the most recent callback invocation.

**Static registry:** `Breakpoint.list_of_breakpoints: Dict[str, List[Breakpoint]]` — every instance is auto-registered by name.

#### `BreakpointController` — Hook Manager

Orchestrates breakpoint attachment/detachment across a model.

```python
ctrl = BreakpointController()
ctrl.add_breakpoint(root=model, target="encoder.conv1", bp=bp, position="after")
```

**Key methods:**
| Method | Description |
|--------|-------------|
| `add_breakpoint(root, target, bp, position)` | Resolves target (str name or `nn.Module`), registers PyTorch hook |
| `add_breakpoint_by_name(root, layer_name, bp, position)` | Convenience wrapper |
| `add_breakpoint_by_module(root, module, bp, position)` | Convenience wrapper |
| `gather()` | Collects `(trace.output, trace.valid)` from all registered breakpoints |
| `clear()` | Removes all hook handles |
| `eval()` / `train()` / `to(device)` | Propagate to callbacks that are `nn.Module` instances |
| `save(path)` / `load(root, path)` | Serialize/deserialize breakpoint configuration + re-attach hooks |
| `state_dict()` | Export serializable config (no raw modules or handles) |

**Shared state:** `ctrl.state: Dict[str, Any]` — a mutable dict passed to every breakpoint invocation via `ctx.state`.

**`"before"` hook mutation contract:**
- `callback` returns `BreakpointOutput(output=<tuple>)`
- `run_before` returns `(result.output, kwargs)` to PyTorch — replacing the positional args
- `ctx.kwargs` is the **same dict object** returned to PyTorch; mutating it in-place propagates keyword-arg changes

**`"after"` hook mutation contract:**
- `callback` returns `BreakpointOutput(output=<value>)`
- `run_after` returns `result.output` directly — replacing the module's output

---

### `var.py` — Data Structures

#### `BreakpointContext` — Runtime Context

Passed to every breakpoint callback during the forward pass.

```python
@dataclass
class BreakpointContext:
    name: str               # breakpoint name (e.g. "my_bp.0")
    layer: str              # layer name from named_modules(), e.g. "layer1.0.conv1"
    position: str           # "before" or "after"
    module: nn.Module       # the nn.Module where the hook is registered
    inputs: tuple           # positional arguments passed to the module
    bp_kwargs: dict         # the Breakpoint's own kwargs
    kwargs: dict            # keyword arguments passed to the module
    output: Any = None      # the module's return value (only set for "after" hooks)
    state: Optional[Dict[str, Any]] = None  # shared controller state
```

| Field | `"before"` | `"after"` |
|-------|-----------|----------|
| `output` | `None` | Module's return value |
| `inputs` | Always set (tuple) | Always set (tuple) |
| `kwargs` | Always set (dict) | Always set (dict) |

#### `BreakpointOutput` — Callback Return Value

```python
@dataclass
class BreakpointOutput:
    fn_name: str            # callback identifier (required)
    context: BreakpointContext | None = None
    output: Dict | List[Tensor] | Tensor | None = None  # replacement value
    trace: Dict | List[Tensor] | Tensor | None = None   # diagnostic trace
    valid: bool = False     # whether the callback considers the result valid
```

#### `EndpointSpec` — Endpoint Configuration

Shared configuration dataclass used by both `aggregation.py` and `aggregate.py`.

```python
@dataclass
class EndpointSpec:
    module: Optional[nn.Module] = None     # nn.Module (used by aggregate.py's Router)
    mode: str = "full"                     # "full" (wait for all keys) or "eager" (fire immediately)
    required_keys: Tuple[str, ...] = ()   # keys needed before firing (full mode)
    pack_fn: Optional[Callable] = None     # packs collected dict → (args, kwargs)
    key: str = ""                          # endpoint identifier / key in collected dict
    transform: Optional[Callable] = None   # data transform applied before buffering (eager mode)
    position: str = "after"               # "before" or "after" (target mode injection)
    input_key: Optional[str] = None       # target kwarg name for "before" injection
```

---

## 2. DAG-in-Breakpoint (`hook_dag.py`)

The DAG-aware breakpoint system in `hook_dag.py` embeds data-flow dependencies directly into `Breakpoint` instances. Instead of a separate aggregation controller, each breakpoint can declare `data_sources` (upstream breakpoint names) and the controller resolves them to `data_sinks` at wiring time.

### Architecture

```
BreakpointController
  ├── add_breakpoint() ── adds breakpoints with optional data_sources
  ├── wire() ── resolves data_sources → data_sinks, validates DAG (no cycles)
  ├── _validate_dag() ── DFS tricolour cycle detection
  └── clear() ── calls bp.reset() on each breakpoint

Forward pass:
  Source BP fires → callback(ctx) → pushes result.output to each sink._buffer
  Target BP fires → reads ctx.collected (snapshot of _buffer) → callback(ctx) → injects
```

### Breakpoint DAG Fields

Each `Breakpoint` gains:

| Field | Type | Set by | Purpose |
|-------|------|--------|---------|
| `data_sources` | `List[str]` | User (constructor) | Names of upstream breakpoints this depends on |
| `data_sinks` | `List[Breakpoint]` | `controller.wire()` | Downstream breakpoints that depend on this one |
| `_buffer` | `Dict[str, Any]` | Upstream breakpoints (runtime) | Data pushed from upstream, keyed by breakpoint name |

### BreakpointContext.collected

The `BreakpointContext` gains a `collected: Dict[str, Any]` field — a snapshot of `self._buffer` passed to the callback. Downstream callbacks read `ctx.collected` to access upstream data.

### Wiring

`BreakpointController.wire()` resolves `data_sources` names to `Breakpoint` instances:

1. Builds `bp_by_name` lookup from registered breakpoints
2. For each breakpoint, resolves each source name (exact match first, then global registry fallback)
3. Appends this breakpoint to each upstream's `data_sinks`
4. Runs `_validate_dag()` — DFS cycle detection; raises `ValueError` on back edge
5. Sets `_wired = True`; subsequent calls are no-ops

`__init_dict__` calls `wire()` automatically after adding all breakpoints.

### Data Flow in Forward Pass

Inside `run_before()` / `run_after()` — natural to the forward pass:

1. Build `BreakpointContext` with `collected=dict(self._buffer)` — snapshot of upstream data
2. Call `callback(ctx)` → `BreakpointOutput`
3. Store `self.trace = result`
4. **Push to sinks**: `for sink in self.data_sinks: sink._buffer[self.name] = result.output`
5. Handle mutation / return

### Example Config

```yaml
breakpoints:
  # Source: captures encoder.0 output (non-mutating)
  - layer_name: encoders.0
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_z0
    pos: after

  # Source: captures encoder.1 output
  - layer_name: encoders.1
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: src_z1
    pos: after

  # Aggregation: collects from sources, injects at head
  - layer_name: head
    bp:
      _target_: src.plugins.hook_dag.Breakpoint
      name: reconstructor
      data_sources: [src_z0, src_z1]
      callback:
        _target_: src.plugins.reconstructor.linear.BilinearReconstructor
        ...
      mutate: true
    pos: before
```

### Python Example

```python
from src.plugins.hook_dag import Breakpoint, BreakpointController

ctrl = BreakpointController()

# Source breakpoints
ctrl.add_breakpoint(model, "encoders.0",
    Breakpoint(name="src_z0"), position="after")
ctrl.add_breakpoint(model, "encoders.1",
    Breakpoint(name="src_z1"), position="after")

# Aggregation breakpoint — reads ctx.collected["src_z0.0"] and ctx.collected["src_z1.0"]
class ReconstructorCB(nn.Module):
    def forward(self, ctx):
        z0 = ctx.collected.get("src_z0.0")
        z1 = ctx.collected.get("src_z1.0")
        fused = torch.cat([z0, z1], dim=-1)
        return BreakpointOutput(fn_name="recon", output=(fused,), valid=True)

ctrl.add_breakpoint(model, "head",
    Breakpoint(name="reconstructor", callback=ReconstructorCB(),
               mutate=True, data_sources=["src_z0.0", "src_z1.0"]),
    position="before")

ctrl.wire()  # resolves data_sources → data_sinks, validates DAG

output = model(x)  # forward pass — data flows through DAG automatically
ctrl.clear()        # resets buffers between passes
```

### Serialization

`data_sources` is included in `state_dict()`. `data_sinks` is NOT — it's reconstructed by `wire()` on load. `load()` and `load_from_state_dict()` both call `wire()` after restoring breakpoints.

### `aggregation.py` (DEPRECATED)

The old `HookDAG`/`AggregationController`/`Endpoint`/`AggregationNode` classes in `aggregation.py` are deprecated. Their DAG logic (cycle detection, wiring) now lives in `BreakpointController.wire()` and `_validate_dag()`. The file is kept for backward compatibility.

---

## 3. Router System (`aggregate.py`)

A **separate, request-based** aggregation system using `EndpointSpec` from `var.py`. Unlike `aggregation.py` (which hooks into model layers), the Router receives explicit `push()` calls with a `request_id`.

### `Router(nn.Module)` — Request-Based Endpoint Dispatcher

```python
router = Router({
    "head_a": EndpointSpec(module=MLP(...), mode="full", required_keys=("x", "mask")),
    "head_b": EndpointSpec(module=MLP(...), mode="eager"),
})

router.push("head_a", request_id="r1", x=t1)
router.push("head_a", request_id="r1", mask=t2)  # "full" mode fires when all keys present
# → {"ready": True, "output": Tensor(...), ...}

router.push_broadcast(request_id="r2", x=t3, mask=t4)
# → {"head_a": {..., "ready": True}, "head_b": {..., "ready": True}}
```

Key methods: `push(endpoint, request_id, **inputs)`, `push_broadcast(request_id, endpoints, **inputs)`, `clear(endpoint, request_id)`, `status(endpoint, request_id)`.

Firing modes:
- **`"full"`**: Accumulates inputs by `request_id` until `required_keys` are all present, then fires `pack_fn` → `module(*args, **kwargs)`
- **`"eager"`**: Fires immediately on every `push()` with whatever inputs are provided

### `EndpointWrapper(nn.Module)` — Module + Spec Container

```python
wrapper = EndpointWrapper(net=MLP(...), spec=EndpointSpec(mode="full", required_keys=("x",)))
```

Bundles an `nn.Module` with its `EndpointSpec` so that it can be passed to a `Router` with its configuration intact.

---

## 4. Reconstructor Modules (`reconstructor/`)

Reconstructors are breakpoint callbacks that **reconstruct missing modality latents** from available ones. They implement `forward(ctx: BreakpointContext) -> BreakpointOutput`.

### `linear.py` — BilinearReconstructor & LinearReconstructor

| Class | Description | Parameters |
|-------|-------------|-----------|
| `BilinearReconstructor` | Bidirectional cross-modal reconstruction via bilinear mapping | `z1_dim=16`, `z2_dim=16`, `hidden_dim=24` |
| `LinearReconstructor` | Simpler linear-only variant | `z1_dim`, `z2_dim` |

Both reconstruct **both directions** from a single breakpoint: captures the concatenated latents `z = torch.cat([z1, z2])` and produces `z1_hat_recon, z2_hat_recon`. Used as EKF observation functions — their Jacobians propagate input covariance to output covariance.

`BilinearReconstructor` network structure:
```
z1 → linear1_1 → hidden1 ─┐
                            ├→ ln12 → z2_hat  (z1 reconstructs z2)
z2 → linear2_1 → hidden2 ─┘

z1 → linear1_2 → hidden1 ─┐
                            ├→ ln21 → z1_hat  (z2 reconstructs z1)
z2 → linear2_2 → hidden2 ─┘
```

### `identity.py` — IdentityHook

```python
class IdentityHook(nn.Module):
    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        return BreakpointOutput(fn_name=..., context=ctx,
                                output=ctx.inputs, trace={"input": ctx.inputs})
```

A no-op passthrough callback. Useful as a placeholder or for capturing inputs without transformation.

---

## 5. Uncertainty Heads (`head/`)

Uncertainty heads are breakpoint callbacks that predict **per-sample predictive uncertainty** from fused latents.

### `bayescap.py` — BayesCap Baseline

| Class | Description |
|-------|-------------|
| `BayesCap1D` | Neural network that outputs `(mu, alpha, beta)` for GGD NLL. Alpha (scale) and beta (shape) both come from learnable heads. |
| `BayesCap1DLoss` | GGD NLL loss: `\|y - mu\|^β / α^β + log(α) + lgamma(1/β) - log(β)` |
| `EKFGGDNLLLoss` | Variant where alpha comes from EKF-propagated `sqrt(sigma_pred_sq)` instead of a neural head. Beta is still a single learnable scalar. |

### `ekf.py` — EKFBiModalInferer

Combines EKF-propagated alpha (from covariance) with a learned beta head.

```
z, sigma_pred_sq → alpha = sqrt(sigma_pred_sq)
                 → beta_head(z) → beta
                 → mu_head(z) → mu
Output: (mu, alpha, beta) for GGD NLL
```

### `hessian.py` — Hessian-Based Uncertainty

| Class | Description |
|-------|-------------|
| `HessianBiModalInferer` | Exact 2nd-order Hessian propagation through the reconstructor. Uses `torch.func.hessian` for curvature-aware uncertainty. |
| `DiscreteHessianBiModalInferer` | Finite-difference approximation of the Hessian — more efficient but approximate. |

---

## 6. Uncertainty Propagation (`ekf_propagation.py`, `sigma_z.py`)

### `ekf_propagation.py` — EKF Covariance Propagation

Two propagation modes:

| Function | Description |
|----------|-------------|
| `diag_ekf_propagation` | Diagonal EKF: propagates `(d_z,)` diagonal covariance through reconstructor Jacobian → `(d_z,)` output variance |
| `full_ekf_propagation` | Full-covariance EKF: propagates `(d_z, d_z)` covariance matrix → `(d_z, d_z)` output covariance |

Both use `torch.func.jacrev` to linearize the reconstructor, then propagate covariance via `J Σ_z J^T`.

### `sigma_z.py` — Input Covariance Providers

Three strategies for estimating the input covariance `Σ_z`:

| Class | Strategy | Requires Source Data? | Description |
|-------|----------|----------------------|-------------|
| `SDSigmaZ` | Source-dependent (Mahalanobis) | Yes | Fits Gaussian on source latents; at inference returns `(d_M²(z) / d_z) * Σ_A` — amplified for OOD |
| `GroundTruthSigmaZ` | Monte Carlo / analytical Jacobian | Yes (distribution params) | MC sampling or `torch.func.jacrev` through frozen encoders |
| `BNShiftSigmaZ` | BatchNorm shift score | No | Per-sample BN shift: `((z_l - μ_l)² / σ_l²).mean()` → `s(z) · I` |

**`SDSigmaZ`** is the primary research variant — it uses Ledoit-Wolf shrinkage (`shrinkage=0.1`) toward an isotropic target to bound the condition number of `Σ_A`, plus a diagonal floor (`cov_floor=1e-4`) for numerical stability.

---

## 7. Callback Contract Summary

Every breakpoint callback — whether a reconstructor, uncertainty head, or aggregation endpoint — implements the same contract:

```
Input:  BreakpointContext
Output: BreakpointOutput
```

**Observer callbacks** (`mutate=False`): The `BreakpointOutput.output` field is ignored for mutation. The callback runs as a side effect (capturing data, logging, etc.).

**Mutating callbacks** (`mutate=True`):
- `"before"` position: `BreakpointOutput.output` must be a `tuple` — replaces positional args. Alternatively, mutate `ctx.kwargs` in-place.
- `"after"` position: `BreakpointOutput.output` replaces the layer's return value.

**Callback class pattern**: All callbacks extend `nn.Module` and define `forward(self, ctx: BreakpointContext) -> BreakpointOutput`. This ensures they work with `BreakpointController.eval()`/`.train()`/`.to()` lifecycle methods.

---

## 8. Typical Usage: End-to-End

### Basic Aggregation

```python
from src.plugins.aggregation import (
    AggregationController, AggregationSpec, SourceSpec, TargetSpec,
)
from src.plugins.hook import BreakpointController

# Define aggregation: fuse early and late features
specs = [
    AggregationSpec(
        name="fuse",
        sources=[
            SourceSpec(layer="encoder.conv1", position="after", key="low"),
            SourceSpec(layer="encoder.conv3", position="after", key="high"),
        ],
        target=TargetSpec(layer="decoder.fc", position="before"),
        aggregate_fn=lambda d: torch.cat([d["low"], d["high"]], dim=-1),
    ),
]

ctrl = BreakpointController()
agg = AggregationController(specs)
agg.register(ctrl, model)

output = model(x)
agg.reset()
```

### Custom Breakpoint Callback

```python
from src.plugins.var import BreakpointContext, BreakpointOutput

class MyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Linear(in_dim, 1)

    def forward(self, ctx: BreakpointContext) -> BreakpointOutput:
        z = ctx.output  # captured layer output
        mu = self.net(z)
        return BreakpointOutput(
            fn_name="MyHead.forward",
            context=ctx,
            output=mu,
            trace={"mu": mu},
            valid=True,
        )

bp = Breakpoint(name="my_head", callback=MyHead(32), mutate=True)
ctrl.add_breakpoint(model, "encoder", bp, position="after")
```
