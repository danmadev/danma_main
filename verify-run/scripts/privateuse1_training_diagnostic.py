"""Bounded direct PrivateUse1 (danma:0) three-neuron, 30x4-sample training.

External diagnostic only; no repository code changes. Uses the benchmark
cluster fixture/dataset/constants and an independent CPU PyTorch SGD
reference. Every step's input is a freshly cloned DANMA leaf; the loss is
computed on explicitly CPU-staged output. An instrumented DANMAClient
subclass records raw requests/replies (no compute alteration).
"""

from __future__ import annotations

import json
import math
import secrets
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from danma_torch.client import DANMAClient, DANMAError, DANMATransportError
from danma_torch.layer import DANMALinear
from danma_torch.training_benchmark import (
    _FEATURES,
    _INITIAL_WEIGHTS,
    _INPUT_IDS,
    _NEURON_IDS,
    _TARGETS,
    _ThreeNodeBenchmarkCluster,
)

REPORT: dict = {"steps": [], "assertions": [], "requests": []}
EPOCHS = 30
TOL = 2e-5


def log(step: str, payload: object) -> None:
    print(f"[{step}] {json.dumps(payload, default=str)}", flush=True)
    REPORT["steps"].append({"step": step, "data": payload})


def check(name: str, condition: bool, detail: str) -> None:
    print(f"  ASSERT {name}: {'PASS' if condition else 'FAIL'} ({detail})", flush=True)
    REPORT["assertions"].append({"name": name, "pass": bool(condition), "detail": detail})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


class ObservingClient(DANMAClient):
    """Real request()/inspect()/forward()/backward(); records wire traffic."""

    def __init__(self, host: str, port: int) -> None:
        super().__init__(host, port)
        self.ports = {port}

    def request(self, message: dict) -> dict:
        reply = DANMAClient.request(self, message)
        REPORT["requests"].append({"direction": "out", "payload": message})
        REPORT["requests"].append({"direction": "in", "payload": reply})
        return reply


def unsupported_add_must_fail(x: torch.Tensor, when: str) -> None:
    raised, detail = False, ""
    try:
        _ = x + x
    except Exception as exc:  # noqa: BLE001
        raised = True
        detail = f"{type(exc).__name__}: {exc}"
    log(f"unsupported_add_{when}", {"raised": raised, "error": detail, "device": str(x.device)})
    check(f"add_unsupported_{when}",
          raised and detail.startswith("NotImplementedError:") and "aten::add" in detail,
          detail)


def to_danma_leaf(cpu_vector: torch.Tensor) -> torch.Tensor:
    fresh = cpu_vector.detach().clone()
    leaf = fresh.to("danma:0")
    leaf.requires_grad_(True)
    return leaf


def dataset_loss_danma(model: DANMALinear) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            rows = []
            for features in _FEATURES:
                row_leaf = to_danma_leaf(features)
                rows.append(model(row_leaf).to("cpu"))
            prediction = torch.stack(rows)
            return float(F.mse_loss(prediction, _TARGETS).item())
    finally:
        model.train(was_training)


def main(run_dir: Path) -> int:
    from danma_torch import enable_privateuse1

    # PyTorch 2.8 sizes accelerator autograd queues when its Engine first
    # starts; register DANMA before the first backward in this fresh process.
    enable_privateuse1()
    torch.manual_seed(0)
    node_bin = Path(__import__("os").environ["DANMA_NODE_BIN"])
    reference = torch.nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        reference.weight.copy_(torch.tensor(_INITIAL_WEIGHTS, dtype=torch.float32))
        reference.bias.zero_()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    with _ThreeNodeBenchmarkCluster(node_bin) as cluster:
        client = ObservingClient("127.0.0.1", cluster.ports[0])
        for port in cluster.ports:
            client.ports.add(port)
        routes = client.routes()
        pids = [proc.pid for proc in cluster.processes]
        argvs = [proc.args for proc in cluster.processes]
        log("cluster", {"pids": pids, "argv": argvs, "ports": cluster.ports, "routes": routes})
        check("routes_owners", all(routes.get(str(n)) == o for n, o in zip(_NEURON_IDS, (1, 2, 3))), str(routes))
        check("nodes_alive", all(p.poll() is None for p in cluster.processes), str(pids))

        initial_states = {n: client.inspect(n) for n in _NEURON_IDS}
        log("initial_states", initial_states)
        check("initial_versions_zero", all(int(s["version"]) == 0 for s in initial_states.values()),
              json.dumps({n: s["version"] for n, s in initial_states.items()}))
        for index, neuron in enumerate(_NEURON_IDS):
            expected = _INITIAL_WEIGHTS[index]
            weights = initial_states[neuron]["weights"]
            check(f"initial_weights_{neuron}",
                  abs(float(weights["901"]) - expected[0]) <= 1e-6 and abs(float(weights["902"]) - expected[1]) <= 1e-6
                  and abs(float(initial_states[neuron]["bias"])) <= 1e-6, str(weights))

        model = DANMALinear(client, neuron_ids=_NEURON_IDS, input_ids=_INPUT_IDS, feedback_ttl_ms=3_000)

        probe = to_danma_leaf(_FEATURES[0])
        unsupported_add_must_fail(probe, "before_training")

        initial_loss = dataset_loss_danma(model)
        with torch.no_grad():
            reference_initial_loss = float(F.mse_loss(reference(_FEATURES), _TARGETS).item())
        log("initial_losses", {"danma": initial_loss, "reference": reference_initial_loss})
        check("initial_loss_matches_reference", abs(initial_loss - reference_initial_loss) <= TOL,
              f"{initial_loss} vs {reference_initial_loss}")

        max_grad_deviation = 0.0
        total_steps = EPOCHS * len(_FEATURES)
        for epoch in range(EPOCHS):
            for sample_index, (features, target) in enumerate(zip(_FEATURES, _TARGETS)):
                step_index = epoch * len(_FEATURES) + sample_index
                inspect_step = step_index in (0, total_steps // 2, total_steps - 1)

                leaf = to_danma_leaf(features)
                check_idx = f"s{step_index}"
                if inspect_step:
                    log(f"input_leaf_{check_idx}", {"values": leaf.tolist(), "device": str(leaf.device)})
                check(f"leaf_device_{check_idx}", leaf.device.type == "danma" and not leaf.is_cpu, str(leaf.device))

                reference_leaf = features.detach().clone().requires_grad_(True)
                reference_prediction = reference(reference_leaf)

                prediction = model(leaf)
                check(f"prediction_device_{check_idx}", prediction.device.type == "danma", str(prediction.device))
                prediction_cpu = prediction.to("cpu")
                loss = F.mse_loss(prediction_cpu, target)
                check(f"loss_finite_{check_idx}", bool(torch.isfinite(loss).item()), str(float(loss.item())))
                reference_loss = F.mse_loss(reference_prediction, target)

                loss.backward()
                reference_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                grad_is_danma = leaf.grad is not None and leaf.grad.device.type == "danma"
                check(f"grad_device_{check_idx}", grad_is_danma,
                      str(leaf.grad.device) if leaf.grad is not None else "None")
                if grad_is_danma:
                    grad_cpu = leaf.grad.to("cpu")
                    if inspect_step:
                        log(f"input_grad_{check_idx}", {"values": grad_cpu.tolist(),
                                                        "reference": reference_leaf.grad.tolist()})
                    deviation = float((grad_cpu - reference_leaf.grad).abs().max().item())
                    max_grad_deviation = max(max_grad_deviation, deviation)
                    check(f"grad_matches_reference_{check_idx}", deviation <= TOL,
                          f"max_abs_dev={deviation}")

        final_loss = dataset_loss_danma(model)
        with torch.no_grad():
            reference_final_loss = float(F.mse_loss(reference(_FEATURES), _TARGETS).item())
        log("final_losses", {"danma": final_loss, "reference": reference_final_loss})
        check("final_loss_below_2pct", final_loss < initial_loss * 0.02, f"{final_loss} vs {initial_loss * 0.02}")
        check("final_loss_matches_reference", abs(final_loss - reference_final_loss) <= TOL,
              f"{final_loss} vs {reference_final_loss}")

        log("max_input_grad_deviation", max_grad_deviation)
        check("grad_deviation_within_tolerance", max_grad_deviation <= TOL, str(max_grad_deviation))

        final_states = {}
        parameter_rows = []
        for index, neuron in enumerate(_NEURON_IDS):
            state = client.inspect(neuron)
            final_states[neuron] = state
            actual = [float(state["weights"]["901"]), float(state["weights"]["902"]), float(state["bias"])]
            expected = [float(v) for v in reference.weight[index].detach()] + [
                float(reference.bias[index].detach())]
            diff = max(abs(a - e) for a, e in zip(actual, expected))
            parameter_rows.append({"neuron": neuron, "actual": actual, "reference": expected, "max_abs_diff": diff})
            check(f"final_version_{neuron}", int(state["version"]) == 120, str(state["version"]))
            check(f"final_params_{neuron}", diff < TOL, str(diff))
        log("final_states", {"parameter_rows": parameter_rows})

        # Correlate captured wire EventIDs with remote trace lifecycle.
        requests = REPORT["requests"]
        def replies(kind_target=None):
            return [r["payload"] for r in requests if r["direction"] == "in"]
        out_msgs = [r["payload"] for r in requests if r["direction"] == "out"]
        forwards = [m for m in out_msgs if m.get("kind") == "forward"]
        backwards = [m for m in out_msgs if m.get("kind") == "backward"]
        traces = [m for m in out_msgs if m.get("kind") == "trace"]
        log("wire_counts", {"forward": len(forwards), "backward": len(backwards), "trace": len(traces),
                            "requests_total": len(out_msgs)})
        expected_forwards = EPOCHS * len(_FEATURES) * len(_NEURON_IDS) + 2 * len(_FEATURES) * len(_NEURON_IDS)
        check("forward_count_expected", len(forwards) == expected_forwards,
              f"{len(forwards)} == {EPOCHS}*{len(_FEATURES)}*{len(_NEURON_IDS)} training + "
              f"2*{len(_FEATURES)}*{len(_NEURON_IDS)} loss-evaluation forwards = {expected_forwards}")
        check("backward_count_120", len(backwards) == EPOCHS * len(_FEATURES) * len(_NEURON_IDS),
              str(len(backwards)))
        # Per-(step, neuron) correlation: within one step the layer sends one
        # forward per neuron and, at backward time, one backward per neuron,
        # each reusing exactly that neuron's forward EventID.
        def pair_for(target_neuron, occurrence):
            fwds = [m for m in forwards if m.get("target") == target_neuron]
            bwds = [m for m in backwards if m.get("target") == target_neuron]
            fwd, bwd = fwds[occurrence], bwds[occurrence]
            training_fwds = [m for m in fwds if m.get("expected")]
            return fwd, bwd, len(training_fwds)

        first = pair_for(_NEURON_IDS[0], 0)
        last_index = EPOCHS * len(_FEATURES) - 1
        last = pair_for(_NEURON_IDS[0], last_index)
        log("event_correlation", {
            "first_forward": first[0], "first_backward": first[1],
            "last_forward": last[0], "last_backward": last[1],
            "training_forwards_neuron11": first[2]})
        check("event_ids_pairwise_first_step",
              first[0]["event_id"] == first[1]["event_id"] and bool(first[0]["expected"]),
              f"fwd={first[0]['event_id']} bwd={first[1]['event_id']}")
        check("event_ids_pairwise_last_step",
              last[0]["event_id"] == last[1]["event_id"] and bool(last[0]["expected"]),
              f"fwd={last[0]['event_id']} bwd={last[1]['event_id']}")
        check("distinct_event_ids", first[0]["event_id"] != last[0]["event_id"], "")
        f_response = next(r["payload"] for r in requests[1:] if r["direction"] == "in"
                          and r["payload"].get("kind") == "forward_result")
        log("forward_reply_example", f_response)

        # Protocol exposes trace via {"kind":"trace"} only during an active
        # activation; after backward the trace is evicted (per Rust
        # three_processes.rs:221 and net lib.rs:408). Probe it for the final
        # completed EventID: it must NOT report an active trace.
        active_probe = client.request({"kind": "trace", "target": _NEURON_IDS[0],
                                       "event_id": first_forward["event_id"], "route_hops": 4})
        log("trace_probe_after_completion", active_probe)
        check("completed_trace_absent", active_probe.get("kind") == "trace_result"
              and active_probe.get("trace") is None, json.dumps(active_probe))

        # Live trace observability for a PrivateUse1-generated EventID is not
        # available here: layer.backward() is the only consumer of the EventID,
        # and an in-flight trace exists only between forward and backward of the
        # same step. Interposing a raw trace query between them is impossible
        # without altering the tested autograd Function. Record this bound.
        REPORT["trace_visibility_bound"] = (
            "EventID pairing forward/backward is proven from captured wire traffic; "
            "a live in-flight trace query between forward and backward cannot be issued "
            "by an external wrapper without modifying tested compute, so same-run "
            "trace-content correlation is NOT asserted.")

        unsupported_add_must_fail(probe, "after_training")

        # Post-training inference through the DANMA path must still work.
        model.eval()
        with torch.no_grad():
            post = model(to_danma_leaf(_FEATURES[0])).to("cpu")
        reference.eval()
        with torch.no_grad():
            post_reference = reference(_FEATURES[0])
        post_diff = float((post - post_reference).abs().max().item())
        log("post_training_inference", {"danma": post.tolist(), "reference": post_reference.tolist(),
                                        "max_abs_diff": post_diff})
        check("post_inference_matches_reference", post_diff <= TOL, str(post_diff))

        # Terminate ONLY owned fixture nodes; retained layer must fail closed.
        killed = cluster.processes[0]
        killed_pid = killed.pid
        killed.terminate()
        killed_rc = killed.wait(timeout=5)
        log("node_killed", {"pid": killed_pid, "returncode": killed_rc})
        check("node_terminated", killed.poll() is not None, str(killed_rc))
        failure, detail = False, ""
        try:
            model.eval()
            with torch.no_grad():
                model(to_danma_leaf(_FEATURES[0]))
        except DANMATransportError as exc:
            failure, detail = True, f"DANMATransportError: {exc}"
        except DANMAError as exc:
            failure, detail = True, f"{type(exc).__name__}: {exc}"
        log("call_after_kill", {"raised": failure, "error": detail})
        check("transport_error_after_kill", failure, detail)

        REPORT["final"] = "PASS"
        Path(run_dir, "training-evidence.json").write_text(json.dumps(REPORT, indent=2, default=str))
        print("TRAINING RESULT: PASS", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit("run via addendum_runner.py")
