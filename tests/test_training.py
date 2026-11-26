import unittest
import tempfile
from pathlib import Path

from src.repro import set_seed
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.plots import TrainingMetrics
from src.training import evaluate_model, train_distillation, train_supervised


class TrainingTests(unittest.TestCase):
    def test_evaluation_uses_sample_weighted_metrics(self):
        logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

        loss, accuracy = evaluate_model(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))

        expected_loss = nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(loss, expected_loss)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_training_metrics_capture_resources_and_round_trip(self):
        set_seed(3)
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        model = nn.Linear(2, 2)

        metrics = train_supervised(
            model,
            loader,
            loader,
            epochs=2,
            lr=0.01,
            device=torch.device("cpu"),
        )
        restored = TrainingMetrics.from_dict(metrics.to_dict())

        self.assertGreaterEqual(restored.training_time_seconds, 0)
        self.assertIsNone(restored.peak_cuda_memory_bytes)
        self.assertEqual(restored.total_parameters, 6)
        self.assertEqual(restored.trainable_parameters, 6)
        self.assertEqual(restored.best_val_acc, metrics.best_val_acc)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure_path = Path(temp_dir) / "training_curve.png"
            restored.plot(save_path=figure_path)
            self.assertTrue(figure_path.is_file())

    def test_output_distillation_trains_student_and_freezes_teacher(self):
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        student = nn.Linear(2, 2)
        teacher = nn.Linear(2, 2)

        metrics = train_distillation(
            student,
            teacher,
            loader,
            loader,
            kd_type="output",
            epochs=1,
            lr=0.01,
            temperature=2.0,
            kd_alpha=0.5,
            feature_weight=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(metrics.val_acc), 1)
        self.assertIsNotNone(metrics.training_time_seconds)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))

    def test_evaluation_rejects_empty_loader(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))

        with self.assertRaisesRegex(ValueError, "empty data loader"):
            evaluate_model(nn.Linear(2, 2), loader, nn.CrossEntropyLoss(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
