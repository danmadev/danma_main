"""Registration for the fail-closed DANMA PrivateUse1 device."""

from __future__ import annotations

import torch

_ENABLED = False


class _DanmaDeviceModule:
    @staticmethod
    def is_available() -> bool:
        from . import _C
        return bool(_C.is_available())

    @staticmethod
    def device_count() -> int:
        from . import _C
        return int(_C.device_count())

    @staticmethod
    def current_device() -> int:
        from . import _C
        return int(_C.current_device())

    @staticmethod
    def set_device(device: object) -> None:
        index = torch.device(device).index
        if index not in (None, 0):
            raise ValueError("DANMA PrivateUse1 exposes only device index 0")

    @staticmethod
    def _is_in_bad_fork() -> bool:
        return False

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        _ = int(seed)


def enable_privateuse1() -> torch.device:
    """Register PrivateUse1 as danma and return torch.device('danma:0').

    The backend deliberately provides no catch-all CPU fallback. Only storage,
    CPU-to-DANMA staging copies, and operators explicitly implemented by DANMA
    are allowed.
    """
    global _ENABLED

    from . import _C  # noqa: F401

    current = torch._C._get_privateuse1_backend_name()
    if current == "privateuseone":
        torch.utils.rename_privateuse1_backend("danma")
    elif current != "danma":
        raise RuntimeError(
            f"PrivateUse1 is already registered as {current!r}; "
            "DANMA cannot share the single PrivateUse1 dispatch key"
        )

    if not _ENABLED:
        if not hasattr(torch, "danma"):
            torch._register_device_module("danma", _DanmaDeviceModule)
        try:
            torch.utils.generate_methods_for_privateuse1_backend(
                for_tensor=True,
                for_module=False,
                for_storage=False,
            )
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
        _ENABLED = True

    return torch.device("danma:0")


def privateuse1_stats() -> dict[str, int | bool | str]:
    """Return proof-oriented backend diagnostics."""
    enable_privateuse1()
    from . import _C

    return {
        "device": str(torch.device("danma:0")),
        "cpu_fallback": False,
        "allocations": int(_C.allocation_count()),
        "copies": int(_C.copy_count()),
    }
