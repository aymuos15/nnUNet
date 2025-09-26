import os
import subprocess
import shutil
from pathlib import Path
import pytest
import nibabel as nib


class TestInference:
    """Test nnUNet inference pipeline using existing trained model."""

    def test_inference_single_image(self, setup_nnunet_paths, dataset_id, temp_output_dir):
        """Test inference on a single test image."""
        # Change to nnUNet directory
        original_cwd = os.getcwd()
        os.chdir('/home/localssk23/nnn/nnUNet')

        try:
            # Get test image
            raw_path = Path(os.environ["nnUNet_raw"])
            test_dir = raw_path / f"Dataset{dataset_id:03d}_Hippocampus" / "imagesTs"
            test_files = list(test_dir.glob("*.nii.gz"))

            assert len(test_files) > 0, "No test images found"

            # Copy first test image to temp input directory
            input_dir = Path(temp_output_dir) / "input"
            output_dir = Path(temp_output_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            test_image = test_files[0]
            shutil.copy2(test_image, input_dir)

            # Check that the existing model exists before inference
            results_path = Path(os.environ["nnUNet_results"])
            model_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__2d"
            assert model_path.exists(), f"Existing model {model_path} not found for inference"

            # Run inference using existing trained model
            cmd = [
                "nnUNetv2_predict",
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-d", str(dataset_id),
                "-c", "2d",
                "-f", "0",
                "-tr", "nnUNetTrainer_1epoch",
                "--disable_tta"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            assert result.returncode == 0, f"Inference failed: {result.stderr}"

            # Check what files were actually created
            created_files = list(output_dir.glob("*.nii.gz"))

            # Expected output file name (nnUNet removes _0000 suffix)
            base_name = test_image.name.replace('_0000.nii.gz', '.nii.gz')
            expected_output = output_dir / base_name

            if not expected_output.exists():
                # Debug: show what files were created
                print(f"Expected: {expected_output}")
                print(f"Created files: {created_files}")

                # If files were created but with different names, use the first one
                if created_files:
                    expected_output = created_files[0]
                    print(f"Using created file: {expected_output}")
                else:
                    assert False, f"No output files created in {output_dir}"

            # Load and check output
            output_img = nib.load(expected_output)
            output_data = output_img.get_fdata()

            # Basic checks on output
            assert output_data.size > 0, "Output image is empty"
            assert output_data.ndim == 3, "Output should be 3D"

        finally:
            os.chdir(original_cwd)

    def test_inference_output_validity(self, setup_nnunet_paths, dataset_id, temp_output_dir):
        """Test that inference output is a valid segmentation."""
        # Change to nnUNet directory
        original_cwd = os.getcwd()
        os.chdir('/home/localssk23/nnn/nnUNet')

        try:
            # Setup same as above
            raw_path = Path(os.environ["nnUNet_raw"])
            test_dir = raw_path / f"Dataset{dataset_id:03d}_Hippocampus" / "imagesTs"
            test_files = list(test_dir.glob("*.nii.gz"))

            input_dir = Path(temp_output_dir) / "input"
            output_dir = Path(temp_output_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            test_image = test_files[0]
            shutil.copy2(test_image, input_dir)

            # Check that the existing model exists before inference
            results_path = Path(os.environ["nnUNet_results"])
            model_path = results_path / f"Dataset{dataset_id:03d}_Hippocampus" / "nnUNetTrainer_1epoch__nnUNetPlans__2d"
            assert model_path.exists(), f"Existing model {model_path} not found for inference"

            # Run inference using existing trained model
            cmd = [
                "nnUNetv2_predict",
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-d", str(dataset_id),
                "-c", "2d",
                "-f", "0",
                "-tr", "nnUNetTrainer_1epoch",
                "--disable_tta"
            ]

            subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Find the actual output file (nnUNet removes _0000 suffix)
            created_files = list(output_dir.glob("*.nii.gz"))
            assert len(created_files) > 0, f"No output files created in {output_dir}"

            output_file = created_files[0]  # Use the first (and likely only) output file
            output_img = nib.load(output_file)
            output_data = output_img.get_fdata()

            # Check segmentation values are valid integers
            unique_values = set(output_data.flatten())
            for val in unique_values:
                assert val >= 0, f"Negative segmentation value found: {val}"
                assert val == int(val), f"Non-integer segmentation value found: {val}"

            # Check that we have background (0) and at least one foreground class
            assert 0 in unique_values, "Background class (0) not found in segmentation"
            assert len(unique_values) > 1, "Only background found - no segmentation performed"

        finally:
            os.chdir(original_cwd)