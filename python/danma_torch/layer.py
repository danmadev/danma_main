"""Autograd bridge for a remote DANMA layer.

DANMA may be driven either by ordinary CPU tensors or by the fail-closed
PrivateUse1 device registered as danma. PrivateUse1 storage is host staging
memory only: affine forward/backward math is performed by remote DANMA neurons,
never by an ATen CPU linear fallback.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Sequence

import torch

from .client import DANMAClient, DANMAError, checked_id


def _event_id() -> int:
    return secrets.randbits(64) or 1


def _is_danma_device(device: torch.device) -> bool:
    return device.type in {"danma", "privateuseone"}


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu":
        return tensor
    if _is_danma_device(tensor.device):
        return tensor.to("cpu")
    raise ValueError(f"unsupported DANMA tensor device: {tensor.device}")


def _from_host(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return tensor
    if _is_danma_device(device):
        return tensor.to(device)
    raise ValueError(f"unsupported DANMA tensor device: {device}")


class _RemoteAffine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        autograd_token: torch.Tensor,
        client: DANMAClient,
        neuron_ids: tuple[int, ...],
        input_ids: tuple[int, ...],
        feedback_ttl_ms: int,
        route_hops: int,
        training: bool,
    ) -> torch.Tensor:
        host_inputs = _to_host(inputs)
        vector_input = host_inputs.ndim == 1
        rows = [host_inputs.detach().tolist()] if vector_input else host_inputs.detach().tolist()
        trace_id = _event_id()
        event_ids: list[list[int]] = []
        result_rows: list[list[float]] = []

        for row in rows:
            outputs: list[float] = []
            events: list[int] = []
            for neuron_id in neuron_ids:
                event_id = _event_id()
                value = client.forward(
                    neuron_id=neuron_id,
                    event_id=event_id,
                    trace_id=trace_id,
                    inputs=list(zip(input_ids, row)),
                    training=training,
                    route_hops=route_hops,
                )
                outputs.append(value)
                events.append(event_id)
            event_ids.append(events)
            result_rows.append(outputs)

        ctx.client = client
        ctx.neuron_ids = neuron_ids
        ctx.input_ids = input_ids
        ctx.event_ids = event_ids
        ctx.training = training
        ctx.feedback_ttl_ms = feedback_ttl_ms
        ctx.route_hops = route_hops
        ctx.vector_input = vector_input
        ctx.input_device = inputs.device
        ctx.consumed = False

        host_output = torch.tensor(result_rows, dtype=torch.float32, device="cpu")
        if vector_input:
            host_output = host_output[0]
        return _from_host(host_output, inputs.device)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple:
        if ctx.consumed:
            raise DANMAError(
                "DANMA EventID already trained by this autograd graph; "
                "a second backward pass cannot update the same activation"
            )
        if not ctx.training:
            raise DANMAError(
                "DANMA inference forward has no saved activation trace; "
                "call module.train() before a differentiable forward"
            )

        host_gradient = _to_host(grad_output)
        expected_shape = (
            (len(ctx.neuron_ids),)
            if ctx.vector_input
            else (len(ctx.event_ids), len(ctx.neuron_ids))
        )
        if tuple(host_gradient.shape) != expected_shape:
            raise DANMAError("PyTorch backward gradient shape does not match DANMA output")
        if host_gradient.dtype != torch.float32:
            raise DANMAError("DANMA backward only supports float32 gradients")
        if not bool(torch.isfinite(host_gradient).all().item()):
            raise DANMAError("DANMA backward received a non-finite gradient")

        ctx.consumed = True
        gradient_rows = (
            [host_gradient.detach().tolist()]
            if ctx.vector_input
            else host_gradient.detach().tolist()
        )
        input_gradients: list[list[float]] = []

        for events, gradients in zip(ctx.event_ids, gradient_rows):
            accumulated = [0.0] * len(ctx.input_ids)
            for neuron_id, event_id, gradient in zip(ctx.neuron_ids, events, gradients):
                contribution = ctx.client.backward(
                    neuron_id=neuron_id,
                    event_id=event_id,
                    gradient=float(gradient),
                    input_ids=ctx.input_ids,
                    feedback_ttl_ms=ctx.feedback_ttl_ms,
                    route_hops=ctx.route_hops,
                )
                for index, value in enumerate(contribution):
                    accumulated[index] += value
            if not all(math.isfinite(value) for value in accumulated):
                raise DANMAError("accumulated DANMA input gradient is not finite")
            input_gradients.append(accumulated)

        host_result = torch.tensor(input_gradients, dtype=torch.float32, device="cpu")
        if ctx.vector_input:
            host_result = host_result[0]
        result = _from_host(host_result, ctx.input_device)
        return result, None, None, None, None, None, None, None


class _RemoteShardAffine(torch.autograd.Function):
    """Local-shard batched affine bridge.

    EventIDs, versions and learning effects remain per neuron. The only shared
    unit is one local transport/mailbox envelope for a compatible neuron set.
    """

    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        autograd_token: torch.Tensor,
        client: DANMAClient,
        neuron_ids: tuple[int, ...],
        input_ids: tuple[int, ...],
        feedback_ttl_ms: int,
        route_hops: int,
        training: bool,
    ) -> torch.Tensor:
        _ = route_hops  # shard RPC is intentionally local-only in this version
        host_inputs = _to_host(inputs)
        vector_input = host_inputs.ndim == 1
        rows = [host_inputs.detach().tolist()] if vector_input else host_inputs.detach().tolist()
        event_ids: list[list[int]] = []
        result_rows: list[list[float]] = []

        for row in rows:
            events = [_event_id() for _ in neuron_ids]
            outputs = client.forward_shard(
                neuron_ids=neuron_ids,
                event_ids=events,
                trace_id=_event_id(),
                inputs=list(zip(input_ids, row)),
                training=training,
            )
            event_ids.append(events)
            result_rows.append(outputs)

        ctx.client = client
        ctx.neuron_ids = neuron_ids
        ctx.input_ids = input_ids
        ctx.event_ids = event_ids
        ctx.training = training
        ctx.feedback_ttl_ms = feedback_ttl_ms
        ctx.vector_input = vector_input
        ctx.input_device = inputs.device
        ctx.consumed = False

        host_output = torch.tensor(result_rows, dtype=torch.float32, device="cpu")
        if vector_input:
            host_output = host_output[0]
        return _from_host(host_output, inputs.device)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple:
        if ctx.consumed:
            raise DANMAError(
                "DANMA shard EventIDs already trained by this autograd graph; "
                "a second backward pass cannot update the same activations"
            )
        if not ctx.training:
            raise DANMAError(
                "DANMA shard inference forward has no saved activation trace"
            )

        host_gradient = _to_host(grad_output)
        expected_shape = (
            (len(ctx.neuron_ids),)
            if ctx.vector_input
            else (len(ctx.event_ids), len(ctx.neuron_ids))
        )
        if tuple(host_gradient.shape) != expected_shape:
            raise DANMAError("PyTorch shard backward gradient shape mismatch")
        if host_gradient.dtype != torch.float32:
            raise DANMAError("DANMA shard backward only supports float32 gradients")
        if not bool(torch.isfinite(host_gradient).all().item()):
            raise DANMAError("DANMA shard backward received a non-finite gradient")

        ctx.consumed = True
        gradient_rows = (
            [host_gradient.detach().tolist()]
            if ctx.vector_input
            else host_gradient.detach().tolist()
        )
        input_gradients: list[list[float]] = []
        for events, gradients in zip(ctx.event_ids, gradient_rows):
            contribution = ctx.client.backward_shard(
                neuron_ids=ctx.neuron_ids,
                event_ids=events,
                gradients=[float(value) for value in gradients],
                input_ids=ctx.input_ids,
                feedback_ttl_ms=ctx.feedback_ttl_ms,
            )
            if not all(math.isfinite(value) for value in contribution):
                raise DANMAError("DANMA shard input gradient is not finite")
            input_gradients.append(contribution)

        host_result = torch.tensor(input_gradients, dtype=torch.float32, device="cpu")
        if ctx.vector_input:
            host_result = host_result[0]
        result = _from_host(host_result, ctx.input_device)
        return result, None, None, None, None, None, None, None


class DANMALinear(torch.nn.Module):
    """A remotely trained affine layer backed by DANMA neuron IDs."""

    def __init__(
        self,
        client: DANMAClient,
        *,
        neuron_ids: Sequence[int],
        input_ids: Sequence[int],
        feedback_ttl_ms: int = 3_000,
        route_hops: int = 4,
        max_batch: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(client, DANMAClient):
            raise TypeError("client must be a DANMAClient")
        self.neuron_ids = tuple(checked_id(value, "neuron_id") for value in neuron_ids)
        self.input_ids = tuple(checked_id(value, "input_id") for value in input_ids)
        if not self.neuron_ids or not self.input_ids:
            raise ValueError("neuron_ids and input_ids must be nonempty")
        if len(set(self.neuron_ids)) != len(self.neuron_ids):
            raise ValueError("neuron_ids must be unique")
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("input_ids must be unique")
        if set(self.neuron_ids) & set(self.input_ids):
            raise ValueError("input_ids and neuron_ids must be disjoint")
        if len(self.neuron_ids) > 1_024 or len(self.input_ids) > 1_024:
            raise ValueError("v1 supports at most 1024 input and output features")
        if type(feedback_ttl_ms) is not int or not 1 <= feedback_ttl_ms <= 10_000:
            raise ValueError("feedback_ttl_ms must be 1..10000")
        if type(route_hops) is not int or not 1 <= route_hops <= 255:
            raise ValueError("route_hops must be 1..255")
        if type(max_batch) is not int or not 1 <= max_batch <= 9:
            raise ValueError("max_batch must be 1..9 (remote staleness budget)")

        self.client = client
        self.feedback_ttl_ms = feedback_ttl_ms
        self.route_hops = route_hops
        self.max_batch = max_batch
        self.register_buffer(
            "_autograd_trigger",
            torch.zeros((), dtype=torch.float32, requires_grad=True),
            persistent=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("DANMALinear requires a torch.Tensor")
        if inputs.device.type != "cpu" and not _is_danma_device(inputs.device):
            raise ValueError("DANMALinear supports only CPU or DANMA tensors")
        if self._autograd_trigger.device.type != "cpu":
            raise ValueError("DANMA autograd trigger must remain on CPU")
        if inputs.dtype != torch.float32:
            raise ValueError("DANMALinear requires float32 inputs")
        if inputs.layout != torch.strided:
            raise ValueError("DANMALinear requires a dense strided input")
        if inputs.ndim not in (1, 2):
            raise ValueError("DANMALinear requires rank 1 or rank 2 input")
        if inputs.shape[-1] != len(self.input_ids):
            raise ValueError("DANMALinear input feature dimension mismatch")
        if inputs.ndim == 2 and not 1 <= inputs.shape[0] <= self.max_batch:
            raise ValueError("DANMALinear batch is empty or exceeds max_batch")

        host_inputs = _to_host(inputs)
        if not bool(torch.isfinite(host_inputs).all().item()):
            raise ValueError("DANMALinear requires finite input values")

        train_remote = bool(self.training and torch.is_grad_enabled())
        token = self._autograd_trigger if train_remote else self._autograd_trigger.detach()
        return _RemoteAffine.apply(
            inputs,
            token,
            self.client,
            self.neuron_ids,
            self.input_ids,
            self.feedback_ttl_ms,
            self.route_hops,
            train_remote,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={len(self.input_ids)}, out_features={len(self.neuron_ids)}, "
            f"remote=True, feedback_ttl_ms={self.feedback_ttl_ms}"
        )


class DANMAShardLinear(DANMALinear):
    """DANMALinear using one local shard RPC per input row.

    All neuron_ids must be owned by the node addressed by client. This
    experimental executor preserves independent neuron state and feedback
    semantics while reducing physical transport granularity.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("DANMAShardLinear requires a torch.Tensor")
        if inputs.device.type != "cpu" and not _is_danma_device(inputs.device):
            raise ValueError("DANMAShardLinear supports only CPU or DANMA tensors")
        if self._autograd_trigger.device.type != "cpu":
            raise ValueError("DANMA autograd trigger must remain on CPU")
        if inputs.dtype != torch.float32:
            raise ValueError("DANMAShardLinear requires float32 inputs")
        if inputs.layout != torch.strided:
            raise ValueError("DANMAShardLinear requires a dense strided input")
        if inputs.ndim not in (1, 2):
            raise ValueError("DANMAShardLinear requires rank 1 or rank 2 input")
        if inputs.shape[-1] != len(self.input_ids):
            raise ValueError("DANMAShardLinear input feature dimension mismatch")
        if inputs.ndim == 2 and not 1 <= inputs.shape[0] <= self.max_batch:
            raise ValueError("DANMAShardLinear batch is empty or exceeds max_batch")

        host_inputs = _to_host(inputs)
        if not bool(torch.isfinite(host_inputs).all().item()):
            raise ValueError("DANMAShardLinear requires finite input values")

        train_remote = bool(self.training and torch.is_grad_enabled())
        token = self._autograd_trigger if train_remote else self._autograd_trigger.detach()
        return _RemoteShardAffine.apply(
            inputs,
            token,
            self.client,
            self.neuron_ids,
            self.input_ids,
            self.feedback_ttl_ms,
            self.route_hops,
            train_remote,
        )
