#!/usr/bin/env python3
"""Manual-equivalent DANMA PrivateUse1 proof harness (external to repository).

Launches one real Rust danma-node, drives forward/backward on danma:0,
verifies remote state changes, unsupported-op failure, inference value,
then terminates ONLY its own node and asserts a transport failure while
the Python layer/client/tensor remain alive. Numeric tolerance 1e-6.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

TOL = 1e-6
NODE = Path(os.environ["DANMA_NODE_BIN"])
REPORT: dict = {"steps": [], "assertions": []}


def log(step: str, payload: object) -> None:
    entry = {"step": step, "data": payload}
    REPORT["steps"].append(entry)
    print(f"[{step}] {json.dumps(payload, default=str)}", flush=True)


def check(name: str, condition: bool, detail: str) -> None:
    entry = {"name": name, "pass": bool(condition), "detail": detail}
    REPORT["assertions"].append(entry)
    print(f"  ASSERT {name}: {'PASS' if condition else 'FAIL'} ({detail})", flush=True)
    if not condition:
        REPORT["final"] = "FAIL"
        Path(os.environ["HARNESS_REPORT"]).write_text(json.dumps(REPORT, indent=2, default=str))
        raise AssertionError(f"{name}: {detail}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    import torch
    from danma_torch import (
        DANMAClient,
        DANMAError,
        DANMALinear,
        DANMATransportError,
        enable_privateuse1,
        privateuse1_stats,
    )

    REPORT["torch"] = torch.__version__
    REPORT["final"] = "PENDING"
    port = free_port()
    proc = subprocess.Popen(
        [
            str(NODE),
            "--id", "1",
            "--listen", f"127.0.0.1:{port}",
            "--neuron", "11",
            "--weight", "901:2",
            "--weight", "902:3",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log("node_launch", {"argv": proc.args, "pid": proc.pid})
    client = DANMAClient("127.0.0.1", port)
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("node exited during startup")
            try:
                if client.inspect(11)["version"] == 0:
                    break
            except Exception:
                time.sleep(0.05)
        else:
            raise TimeoutError("node not ready")

        enable_privateuse1()
        device = torch.device("danma:0")
        stats_before = dict(privateuse1_stats())
        log("stats_before", stats_before)

        x = torch.tensor([1.0, 2.0], dtype=torch.float32).to(device)
        x.requires_grad_(True)
        log("tensor_x", {"device": str(x.device), "is_cpu": bool(x.is_cpu), "values": x.tolist(), "requires_grad": x.requires_grad})
        check("x_on_danma", x.device.type == "danma", str(x.device))
        check("x_not_cpu", not x.is_cpu, f"is_cpu={x.is_cpu}")

        try:
            _ = x + x
            unsupported_raised = False
            err = "no exception"
        except Exception as exc:  # noqa: BLE001
            unsupported_raised = True
            err = f"{type(exc).__name__}: {exc}"
        log("unsupported_x_plus_x", {"raised": unsupported_raised, "error": err})
        check("x_plus_x_raises", unsupported_raised, err)

        inspect_before = client.inspect(11)
        log("inspect_before", inspect_before)
        check("version_before_0", int(inspect_before["version"]) == 0, f"version={inspect_before['version']}")

        layer = DANMALinear(client, neuron_ids=(11,), input_ids=(901, 902))
        y = layer(x)
        y_host_value = float(y.to("cpu").item())
        log("forward_y", {"value_host_read_via_cpu_copy": y_host_value, "device": str(y.device), "is_cpu": bool(y.is_cpu), "note": "aten::_local_scalar_dense unsupported on danma; value read after explicit CPU copy"})
        check("y_value_8", abs(y_host_value - 8.0) <= TOL, f"value={y_host_value}")
        check("y_on_danma", y.device.type == "danma", str(y.device))

        loss = y.to("cpu").sum()
        log("loss", {"value": float(loss.item()), "device": str(loss.device)})
        check("loss_8", abs(float(loss.item()) - 8.0) <= TOL, f"loss={float(loss.item())}")
        loss.backward()

        inspect_after = client.inspect(11)
        log("inspect_after", inspect_after)
        check("version_after_1", int(inspect_after["version"]) == 1, f"version={inspect_after['version']}")
        w901 = float(inspect_after["weights"]["901"])
        w902 = float(inspect_after["weights"]["902"])
        bias = float(inspect_after["bias"])
        check("weight_901_1_9", abs(w901 - 1.9) <= TOL, f"w901={w901}")
        check("weight_902_2_8", abs(w902 - 2.8) <= TOL, f"w902={w902}")
        check("bias_minus_0_1", abs(bias - (-0.1)) <= TOL, f"bias={bias}")

        grads = x.grad.tolist() if x.grad is not None else None
        log("x_grad", {"values": grads, "device": str(x.grad.device) if x.grad is not None else None})
        check("grad_value_2_3", grads is not None and all(abs(a - b) <= TOL for a, b in zip(grads, [2.0, 3.0])), f"grads={grads}")
        check("grad_on_danma", x.grad is not None and x.grad.device.type == "danma", str(x.grad.device) if x.grad is not None else "None")

        stats_after = dict(privateuse1_stats())
        log("stats_after", stats_after)

        layer.eval()
        with torch.no_grad():
            y_inf = layer(x)
        y_inf_host = float(y_inf.to("cpu").item())
        log("inference_after_update", {"value_host_read_via_cpu_copy": y_inf_host, "device": str(y_inf.device)})
        check("inference_7_4", abs(y_inf_host - 7.4) <= TOL, f"value={y_inf_host}")

        log("parameters", {"count": len(list(layer.parameters())), "params": [str(p) for p in layer.parameters()]})
        sd = layer.state_dict()
        log("state_dict", {"keys": list(sd.keys()), "note": "module state_dict is local; remote neuron state is authoritative (inspect_after)"})

        proc.terminate()
        returncode = proc.wait(timeout=5)
        log("node_stopped", {"pid": proc.pid, "returncode": returncode, "alive": proc.poll() is None})
        check("node_terminated", not (proc.poll() is None), f"returncode={returncode}")

        try:
            _ = layer(x)
            transport_raised = False
            terr = "no exception"
        except DANMATransportError as exc:
            transport_raised = True
            terr = f"DANMATransportError: {exc}"
        except DANMAError as exc:
            transport_raised = True
            terr = f"{type(exc).__name__} (DANMAError subclass): {exc}"
        except Exception as exc:  # noqa: BLE001
            transport_raised = False
            terr = f"unexpected {type(exc).__name__}: {exc}"
        log("call_after_kill", {"raised": transport_raised, "error": terr, "layer_alive": True, "client_alive": True})
        check("transport_error_after_kill", transport_raised, terr)

        REPORT["final"] = "PASS"
        Path(os.environ["HARNESS_REPORT"]).write_text(json.dumps(REPORT, indent=2, default=str))
        print("HARNESS RESULT: PASS", flush=True)
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        print(f"[cleanup] node pid {proc.pid} final state returncode={proc.returncode}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
