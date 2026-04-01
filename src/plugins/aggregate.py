import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple, Callable, Sequence
from src.plugins.var import EndpointSpec

def default_pack_fn(inputs: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Mặc định: truyền dạng keyword arguments cho module(**inputs).
    """
    return (), dict(inputs)

class Node(nn.Module):
    """
    Node tính toán chung 1 computational graph, lấy ý tưởng tương tự GStreamer
    Bao gồm: 
    srcs: địa chỉ 
    """
    
class Router(nn.Module):
    """
    Router gom inputs cho các endpoint.
    - Mỗi endpoint có mode: full/eager
    - full: tích lũy inputs theo request_id, đủ required_keys -> chạy module
    - eager: chạy module ngay với inputs hiện có (không tích lũy)
    """

    def __init__(self, endpoints: Dict[str, EndpointSpec]):
        super().__init__()

        # ModuleDict để PyTorch quản lý params
        self.endpoints = nn.ModuleDict({name: spec.module for name, spec in endpoints.items()})

        # Lưu metadata/spec riêng
        self._specs: Dict[str, EndpointSpec] = {}
        for name, spec in endpoints.items():
            if spec.mode not in ("full", "eager"):
                raise ValueError(f"Endpoint '{name}' mode must be 'full' or 'eager', got {spec.mode}")
            if spec.pack_fn is None:
                spec.pack_fn = default_pack_fn
            self._specs[name] = spec

        # pending cache: pending[endpoint][request_id] = dict(inputs)
        self._pending: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name in endpoints.keys()}

    @torch.no_grad()
    def clear(self, endpoint: Optional[str] = None, request_id: Optional[str] = None):
        """
        Xóa cache pending.
        - clear()                          -> xóa tất cả
        - clear(endpoint="A")              -> xóa all request của endpoint A
        - clear(endpoint="A", request_id="r1") -> xóa riêng request r1
        """
        if endpoint is None:
            for ep in self._pending:
                self._pending[ep].clear()
            return

        if endpoint not in self._pending:
            raise KeyError(f"Unknown endpoint '{endpoint}'")

        if request_id is None:
            self._pending[endpoint].clear()
        else:
            self._pending[endpoint].pop(request_id, None)

    def status(self, endpoint: str, request_id: str) -> Dict[str, Any]:
        """
        Trả về trạng thái pending cho debug.
        """
        if endpoint not in self._specs:
            raise KeyError(f"Unknown endpoint '{endpoint}'")

        spec = self._specs[endpoint]
        got = set(self._pending[endpoint].get(request_id, {}).keys())
        need = set(spec.required_keys)

        return {
            "endpoint": endpoint,
            "request_id": request_id,
            "mode": spec.mode,
            "got_keys": sorted(got),
            "required_keys": list(spec.required_keys),
            "missing_keys": sorted(list(need - got)),
            "ready": need.issubset(got) if spec.mode == "full" else True,
        }

    def push(
        self,
        endpoint: str,
        request_id: str,
        **inputs: Any,
    ) -> Dict[str, Any]:
        """
        Đẩy một phần inputs vào endpoint.
        Trả về dict:
          {
            "ready": bool,
            "output": tensor/any or None,
            "used_inputs": dict (nếu chạy),
            "cached_keys": list[str],
          }
        """
        if endpoint not in self._specs:
            raise KeyError(f"Unknown endpoint '{endpoint}'")

        spec = self._specs[endpoint]
        module = self.endpoints[endpoint]

        if spec.mode == "eager":
            # EAGER: chạy ngay với inputs hiện có
            args, kwargs = spec.pack_fn(dict(inputs))
            out = module(*args, **kwargs)
            return {
                "ready": True,
                "output": out,
                "used_inputs": dict(inputs),
                "cached_keys": [],
            }

        # FULL: cache theo request_id
        pend = self._pending[endpoint].setdefault(request_id, {})
        pend.update(inputs)

        # check đủ required keys chưa
        required = set(spec.required_keys)
        got = set(pend.keys())
        ready = required.issubset(got)

        if not ready:
            return {
                "ready": False,
                "output": None,
                "used_inputs": None,
                "cached_keys": sorted(list(got)),
            }

        # đủ -> chạy module và clear request cache
        used_inputs = {k: pend[k] for k in spec.required_keys}
        args, kwargs = spec.pack_fn(used_inputs)
        out = module(*args, **kwargs)

        # clear only this request_id for endpoint
        self._pending[endpoint].pop(request_id, None)

        return {
            "ready": True,
            "output": out,
            "used_inputs": used_inputs,
            "cached_keys": [],
        }
    
    def push_broadcast(
        self,
        request_id: str,
        endpoints: Optional[Sequence[str]] = None,
        **inputs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Broadcast cùng một inputs tới nhiều endpoint.

        Args:
          request_id: id của request (full mode sẽ cache theo id này)
          endpoints: list endpoint names. Nếu None -> broadcast tới tất cả endpoint.
          **inputs: payload chung (vd x=..., t=..., mask=...)

        Returns:
          dict: {endpoint_name: push_result_dict}
        """
        if endpoints is None:
            endpoints = list(self._specs.keys())

        results: Dict[str, Dict[str, Any]] = {}
        for ep in endpoints:
            results[ep] = self.push(ep, request_id, **inputs)
        return results

class EndpointWrapper(nn.Module):
    def __init__(self,
                 net: nn.Module,  
                 spec: EndpointSpec | None = None, 
                 ):
        super().__init__()
        # Contain spec to pass on Router as requirements
        self.spec = spec if spec is not None else EndpointSpec()
        # Contained net
        self.net = net
    
    def forward(self, **kwargs):
        # Digest aggregated input and produce output 
        args = self.spec.pack_fn(kwargs)
        return self.net(args)
