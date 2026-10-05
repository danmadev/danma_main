#!/usr/bin/env python3
"""External diagnostics: per-neuron final Rust weights/bias vs PyTorch reference.

Reuses the repository's own training_benchmark module (cluster, data, loop
semantics) without modifying it. Fresh process; real three-process cluster.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from danma_torch.training_benchmark import (
    _FEATURES,
    _INITIAL_WEIGHTS,
    _INPUT_IDS,
    _NEURON_IDS,
    _TARGETS,
    _ThreeNodeBenchmarkCluster,
)
from danma_torch.layer import DANMALinear

NODE = Path(os.environ["DANMA_NODE_BIN"])


def main() -> int:
    torch.manual_seed(0)
    reference = torch.nn.Linear(2, 3, bias=True)
    with torch.no_grad():
        reference.weight.copy_(torch.tensor(_INITIAL_WEIGHTS, dtype=torch.float32))
        reference.bias.zero_()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    epochs = 30
    out: dict = {"epochs": epochs, "input_tensors_device": "cpu (adapter serializes to Rust neurons)", "neurons": []}
    with _ThreeNodeBenchmarkCluster(NODE) as cluster:
        model = DANMALinear(cluster.client, neuron_ids=_NEURON_IDS, input_ids=_INPUT_IDS, feedback_ttl_ms=3_000)
        for _ in range(epochs):
            for features, target in zip(_FEATURES, _TARGETS):
                F.mse_loss(model(features), target).backward()
                optimizer.zero_grad(set_to_none=True)
                ref_loss = F.mse_loss(reference(features), target)
                ref_loss.backward()
                optimizer.step()
        for index, neuron_id in enumerate(_NEURON_IDS):
            state = cluster.client.inspect(neuron_id)
            actual = [
                float(state["weights"][str(_INPUT_IDS[0])]),
                float(state["weights"][str(_INPUT_IDS[1])]),
                float(state["bias"]),
            ]
            expected = [
                float(reference.weight[index][0]),
                float(reference.weight[index][1]),
                float(reference.bias[index]),
            ]
            diffs = [abs(a - e) for a, e in zip(actual, expected)]
            out["neurons"].append(
                {
                    "neuron_id": neuron_id,
                    "version": int(state["version"]),
                    "rust_weights_bias": actual,
                    "torch_reference": expected,
                    "abs_diff": diffs,
                    "max_abs_diff": max(diffs),
                }
            )
        out["routes_via_entry_node1"] = {
            n: cluster.client.routes().get(str(n)) for n in _NEURON_IDS
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
