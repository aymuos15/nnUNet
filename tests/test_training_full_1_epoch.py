import os
import subprocess
import pytest
from pathlib import Path


class TestTrainingFull1Epoch:
    """Test full nnUNet training pipeline for 1 epoch."""

    def test_plan_and_preprocess_with_integrity_check(self, setup_nnunet_paths, dataset_id):
        """Test dataset planning and preprocessing with integrity verification."""
        # Change to nnUNet directory
        original_cwd = os.getcwd()
        os.chdir('/home/localssk23/nnn/nnUNet')

        try:
            # Run planning and preprocessing with integrity check
            cmd = [
                "nnUNetv2_plan_and_preprocess",
                "-d", str(dataset_id),
                "--verify_dataset_integrity"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            assert result.returncode == 0, f"Planning with integrity check failed: {result.stderr}"

            # Check preprocessed files exist
            preprocessed_path = Path(os.environ["nnUNet_preprocessed"])
            dataset_path = preprocessed_path / f"Dataset{dataset_id:03d}_Hippocampus"

            assert dataset_path.exists(), "Preprocessed dataset directory not created"

            # Check specific preprocessed files
            expected_files = [
                "dataset_fingerprint.json",
                "splits_final.json",
                "nnUNetPlans.json"
            ]

            for filename in expected_files:
                file_path = dataset_path / filename
                assert file_path.exists(), f"Expected file {filename} not found in {dataset_path}"

        finally:
            os.chdir(original_cwd)

    def test_training_one_epoch_2d(self, setup_nnunet_paths, dataset_id):
        """Test training 2D configuration for 1 epoch."""
        # Change to nnUNet directory
        original_cwd = os.getcwd()
        os.chdir('/home/localssk23/nnn/nnUNet')

        try:
            # Train 2D configuration for 1 epoch (fastest)
            cmd = [
                "nnUNetv2_train",
                str(dataset_id),
                "2d",
                "0",  # fold 0
                "--npz",
                "-tr", "nnUNetTrainer_1epoch"
            ]

            # Set a longer timeout for training (30 minutes)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            assert result.returncode == 0, f"Training failed: {result.stderr}"

            # Check model files were created
            results_path = Path(os.environ["nnUNet_results"])
            model_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__2d" / "fold_0"

            assert model_path.exists(), "Model directory not created"

            # Check for checkpoint files (at least one should exist)
            checkpoint_patterns = [
                "checkpoint_final.pth",
                "checkpoint_latest.pth",
                "checkpoint_best.pth"
            ]

            checkpoint_found = False
            for pattern in checkpoint_patterns:
                checkpoint_file = model_path / pattern
                if checkpoint_file.exists():
                    checkpoint_found = True
                    print(f"Found checkpoint: {checkpoint_file}")
                    break

            assert checkpoint_found, f"No checkpoint files found in {model_path}. Expected one of: {checkpoint_patterns}"

            # Check training log files
            expected_log_files = [
                "training_log.txt",
                "progress.png"
            ]

            for log_file in expected_log_files:
                log_path = model_path / log_file
                if log_path.exists():
                    print(f"Found log file: {log_path}")

        finally:
            os.chdir(original_cwd)

    def test_training_one_epoch_3d_fullres(self, setup_nnunet_paths, dataset_id):
        """Test training 3D fullres configuration for 1 epoch."""
        # Change to nnUNet directory
        original_cwd = os.getcwd()
        os.chdir('/home/localssk23/nnn/nnUNet')

        try:
            # Train 3D fullres configuration for 1 epoch
            cmd = [
                "nnUNetv2_train",
                str(dataset_id),
                "3d_fullres",
                "0",  # fold 0
                "--npz",
                "-tr", "nnUNetTrainer_1epoch"
            ]

            # Set a longer timeout for 3D training (60 minutes)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            assert result.returncode == 0, f"3D training failed: {result.stderr}"

            # Check model files were created
            results_path = Path(os.environ["nnUNet_results"])
            model_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres" / "fold_0"

            assert model_path.exists(), "3D model directory not created"

            # Check for checkpoint files (at least one should exist)
            checkpoint_patterns = [
                "checkpoint_final.pth",
                "checkpoint_latest.pth",
                "checkpoint_best.pth"
            ]

            checkpoint_found = False
            for pattern in checkpoint_patterns:
                checkpoint_file = model_path / pattern
                if checkpoint_file.exists():
                    checkpoint_found = True
                    print(f"Found 3D checkpoint: {checkpoint_file}")
                    break

            assert checkpoint_found, f"No 3D checkpoint files found in {model_path}. Expected one of: {checkpoint_patterns}"

            # Check training log files
            expected_log_files = [
                "training_log.txt",
                "progress.png"
            ]

            for log_file in expected_log_files:
                log_path = model_path / log_file
                if log_path.exists():
                    print(f"Found 3D log file: {log_path}")

        finally:
            os.chdir(original_cwd)

    def test_training_produces_valid_models(self, setup_nnunet_paths, dataset_id):
        """Test that training produces valid models that can be loaded."""
        import json

        results_path = Path(os.environ["nnUNet_results"])

        # Test both 2D and 3D models
        configurations = [
            ("2d", "nnUNetTrainer_1epoch__nnUNetPlans__2d"),
            ("3d_fullres", "nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres")
        ]

        for config_name, model_dir in configurations:
            model_base_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / model_dir

            # Check model directory structure
            assert model_base_path.exists(), f"Model base path {model_base_path} not found"

            # Check required model files
            required_files = [
                "plans.json",
                "dataset.json",
                "dataset_fingerprint.json"
            ]

            for filename in required_files:
                file_path = model_base_path / filename
                assert file_path.exists(), f"Required model file {filename} not found for {config_name}"

            # Validate plans.json structure
            plans_file = model_base_path / "plans.json"
            with open(plans_file, 'r') as f:
                plans_data = json.load(f)

            # Check required fields in plans
            required_plan_fields = ["configurations", "experiment_planner_used", "label_manager"]
            for field in required_plan_fields:
                assert field in plans_data, f"Required field '{field}' missing from plans.json for {config_name}"

            # Check that the configuration exists in plans
            assert config_name in plans_data["configurations"], f"{config_name} configuration not found in plans"

            print(f"Model validation successful for {config_name}: {model_base_path}")