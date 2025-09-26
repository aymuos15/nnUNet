import os
import subprocess
import pytest
from pathlib import Path


class TestTraining:
    """Test nnUNet training pipeline using existing models."""

    def test_existing_model_structure(self, setup_nnunet_paths, dataset_id):
        """Test that existing trained model has correct structure."""
        results_path = Path(os.environ["nnUNet_results"])
        model_base_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus"
        model_path = model_base_path / "nnUNetTrainer_1epoch__nnUNetPlans__2d"

        assert model_base_path.exists(), f"Model base directory {model_base_path} not found"
        assert model_path.exists(), f"Model directory {model_path} not found"

        # Check required files in model directory
        expected_files = [
            "dataset_fingerprint.json",
            "dataset.json",
            "plans.json"
        ]

        for filename in expected_files:
            file_path = model_path / filename
            assert file_path.exists(), f"Expected file {filename} not found in {model_path}"

        # Check fold directory exists
        fold_path = model_path / "fold_0"
        assert fold_path.exists(), f"Fold directory {fold_path} not found"

    def test_model_plans_file_validity(self, setup_nnunet_paths, dataset_id):
        """Test that model plans.json file is valid."""
        import json

        results_path = Path(os.environ["nnUNet_results"])
        plans_file = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__2d" / "plans.json"

        assert plans_file.exists(), f"Plans file {plans_file} not found"

        # Load and validate JSON
        with open(plans_file, 'r') as f:
            plans_data = json.load(f)

        # Check required fields
        required_fields = ["configurations", "experiment_planner_used", "label_manager"]
        for field in required_fields:
            assert field in plans_data, f"Required field '{field}' missing from plans.json"

    def test_existing_checkpoint_files(self, setup_nnunet_paths, dataset_id):
        """Test that existing model has required checkpoint files."""
        results_path = Path(os.environ["nnUNet_results"])
        fold_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__2d" / "fold_0"

        assert fold_path.exists(), f"Fold directory {fold_path} not found"

        # Check for checkpoint files (at least one should exist)
        checkpoint_patterns = [
            "checkpoint_final.pth",
            "checkpoint_latest.pth",
            "checkpoint_best.pth"
        ]

        checkpoint_found = False
        for pattern in checkpoint_patterns:
            checkpoint_file = fold_path / pattern
            if checkpoint_file.exists():
                checkpoint_found = True
                break

        assert checkpoint_found, f"No checkpoint files found in {fold_path}. Expected one of: {checkpoint_patterns}"