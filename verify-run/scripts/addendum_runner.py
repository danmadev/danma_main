"""Fresh-process evidence envelope; no changes to tested source or old harness."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    mode, directory = sys.argv[1:]
    root = Path(directory).resolve()
    output = root / (mode + ".json")
    if output.exists():
        raise FileExistsError(output)
    scripts = {"manual": "manual_harness.py", "training": "privateuse1_training_diagnostic.py",
               "copy": "safe_copy_diagnostic.py"}
    script = Path(__file__).parent / scripts[mode]
    import torch
    import numpy
    import danma_torch
    from danma_torch import _C
    modules = [danma_torch, _C] + [importlib.import_module("danma_torch." + n)
                                for n in ("client", "layer", "backend", "training_benchmark")]
    paths = [__file__, script, sys.executable, os.environ["DANMA_NODE_BIN"]] + [m.__file__ for m in modules]
    result = {"run_id": root.name, "mode": mode, "pid": os.getpid(), "started_utc": stamp(),
              "command": {"argv": sys.argv, "cwd": os.getcwd(), "executable": sys.executable,
                          "DANMA_NODE_BIN": os.environ["DANMA_NODE_BIN"]},
              "python": sys.version, "prefix": sys.prefix, "torch": torch.__version__,
              "numpy": numpy.__version__, "artifacts_before": [fingerprint(p) for p in paths],
              "sha": subprocess.check_output(["git", "-C", "verify-run/danma-clean", "rev-parse", "HEAD"], text=True).strip()}
    print("RUN_START " + json.dumps(result), flush=True)
    start = time.monotonic()
    code = 1
    def bounded(signum, frame):
        raise TimeoutError("external diagnostic exceeded 150-second limit or was terminated")
    signal.signal(signal.SIGALRM, bounded)
    signal.signal(signal.SIGTERM, bounded)
    signal.alarm(150)
    try:
        if mode == "manual":
            os.environ["HARNESS_REPORT"] = str(root / "manual-harness-data.json")
            ns = runpy.run_path(str(script), run_name="external_manual")
            code = ns["main"]()
            result["evidence"] = ns["REPORT"]
            steps = {e["step"]: e["data"] for e in result["evidence"]["steps"]}
            assert steps["tensor_x"]["device"] == steps["forward_y"]["device"] == "danma:0"
            assert steps["x_grad"]["device"] == "danma:0" and steps["x_grad"]["values"] == [2.0, 3.0]
            assert steps["unsupported_x_plus_x"]["error"].startswith("NotImplementedError:")
            assert "aten::add.out" in steps["unsupported_x_plus_x"]["error"]
            assert steps["call_after_kill"]["error"].startswith("DANMATransportError:")
            assert steps["node_launch"]["pid"] == steps["node_stopped"]["pid"]
            result["strict_observation_checks"] = "PASS"
        else:
            ns = runpy.run_path(str(script), run_name="external_diagnostic")
            result["evidence"] = ns["REPORT"]
            code = ns["main"](root)
    except BaseException as exc:
        code = 1
        result["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(traceback.format_exc(), flush=True)
        if "ns" in locals() and "REPORT" in ns:
            result["evidence"] = ns["REPORT"]
    finally:
        signal.alarm(0)
        result.update(ended_utc=stamp(), elapsed_seconds=time.monotonic()-start, exit_code=code,
                      artifacts_after=[fingerprint(p) for p in paths])
        assert result["artifacts_before"] == result["artifacts_after"], "runtime artifact bytes changed"
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print("RUN_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
