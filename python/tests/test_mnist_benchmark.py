from __future__ import annotations

import unittest

import torch

from danma_torch.mnist_benchmark import (
    _balanced,
    _comparisons,
    _orders,
    _run_torch,
    _summary,
)


class MNISTBenchmarkUnitTests(unittest.TestCase):
    def test_balanced_subset_is_class_balanced(self) -> None:
        labels = torch.arange(10, dtype=torch.long).repeat_interleave(4)
        images = torch.arange(40 * 784, dtype=torch.float32).reshape(40, 784)
        subset_images, subset_labels = _balanced(images, labels, 20, seed=5)
        self.assertEqual(tuple(subset_images.shape), (20, 784))
        self.assertEqual(torch.bincount(subset_labels, minlength=10).tolist(), [2] * 10)

    def test_stratified_subset_returns_exact_count_when_classes_are_imbalanced(self) -> None:
        counts = [1, 3] + [2] * 8
        labels = torch.cat([
            torch.full((count,), cls, dtype=torch.long)
            for cls, count in enumerate(counts)
        ])
        images = torch.arange(len(labels) * 784, dtype=torch.float32).reshape(len(labels), 784)
        subset_images, subset_labels = _balanced(
            images, labels, len(labels), seed=11
        )
        self.assertEqual(tuple(subset_images.shape), (len(labels), 784))
        self.assertEqual(len(subset_labels), len(labels))
        self.assertEqual(
            torch.bincount(subset_labels, minlength=10).tolist(),
            counts,
        )

    def test_cpu_backend_reports_quality_and_speed(self) -> None:
        torch.manual_seed(0)
        train_x = torch.rand(16, 784)
        train_y = torch.arange(16, dtype=torch.long) % 10
        test_x = torch.rand(10, 784)
        test_y = torch.arange(10, dtype=torch.long)
        metrics, state = _run_torch(
            "cpu",
            train_x,
            train_y,
            test_x,
            test_y,
            _orders(16, 1, seed=3),
            8,
        )
        self.assertEqual(metrics["status"], "ok")
        self.assertGreater(metrics["train_seconds"], 0.0)
        self.assertGreater(metrics["train_samples_per_second"], 0.0)
        self.assertGreater(metrics["eval_samples_per_second"], 0.0)
        self.assertGreater(metrics["train_batch_latency_seconds"]["p95"], 0.0)
        self.assertGreaterEqual(metrics["test_accuracy"], 0.0)
        self.assertLessEqual(metrics["test_accuracy"], 1.0)
        self.assertEqual(state.numel(), 784 * 10 + 10)

    def test_comparisons_report_speed_and_quality_gaps(self) -> None:
        results = {
            "cpu": {
                "status": "ok",
                "train_samples_per_second": 100.0,
                "eval_samples_per_second": 200.0,
                "test_accuracy": 0.80,
                "test_loss": 0.50,
            },
            "danma": {
                "status": "ok",
                "train_samples_per_second": 25.0,
                "eval_samples_per_second": 50.0,
                "test_accuracy": 0.79,
                "test_loss": 0.51,
            },
        }
        states = {
            "cpu": torch.zeros(3),
            "danma": torch.tensor([0.0, 1e-6, 0.0]),
        }
        result = _comparisons(results, states)["danma"]
        self.assertAlmostEqual(result["train_throughput_ratio_vs_cpu"], 0.25)
        self.assertAlmostEqual(result["eval_throughput_ratio_vs_cpu"], 0.25)
        self.assertAlmostEqual(
            result["test_accuracy_gap_vs_cpu_percentage_points"], -1.0
        )
        self.assertAlmostEqual(result["max_parameter_error_vs_cpu"], 1e-6)

    def test_latency_percentiles(self) -> None:
        result = _summary([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(result["p50"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertEqual(result["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
