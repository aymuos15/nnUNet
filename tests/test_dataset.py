import os
import json
from pathlib import Path
import pytest


class TestDataset:
    """Test dataset verification."""

    def test_dataset_exists(self, setup_nnunet_paths, dataset_id):
        """Test that Dataset004_Hippocampus exists."""
        raw_path = Path(os.environ["nnUNet_raw"])
        dataset_path = raw_path / f"Dataset{dataset_id:03d}_Hippocampus"

        assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"
        assert dataset_path.is_dir(), f"Dataset path {dataset_path} is not a directory"

    def test_dataset_structure(self, setup_nnunet_paths, dataset_id):
        """Test dataset has required directories and files."""
        raw_path = Path(os.environ["nnUNet_raw"])
        dataset_path = raw_path / f"Dataset{dataset_id:03d}_Hippocampus"

        # Check required directories
        required_dirs = ["imagesTr", "labelsTr", "imagesTs"]
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            assert dir_path.exists(), f"Required directory {dir_path} missing"
            assert dir_path.is_dir(), f"{dir_path} is not a directory"

        # Check dataset.json exists
        dataset_json = dataset_path / "dataset.json"
        assert dataset_json.exists(), "dataset.json missing"

    def test_dataset_json_valid(self, setup_nnunet_paths, dataset_id):
        """Test dataset.json is valid JSON with required fields."""
        raw_path = Path(os.environ["nnUNet_raw"])
        dataset_path = raw_path / f"Dataset{dataset_id:03d}_Hippocampus"
        dataset_json = dataset_path / "dataset.json"

        with open(dataset_json, 'r') as f:
            data = json.load(f)

        # Check required fields
        required_fields = ["name", "description", "tensorImageSize", "modality", "labels"]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from dataset.json"

    def test_training_images_exist(self, setup_nnunet_paths, dataset_id):
        """Test training images exist and are paired with labels."""
        raw_path = Path(os.environ["nnUNet_raw"])
        dataset_path = raw_path / f"Dataset{dataset_id:03d}_Hippocampus"

        images_dir = dataset_path / "imagesTr"
        labels_dir = dataset_path / "labelsTr"

        image_files = list(images_dir.glob("*.nii.gz"))
        label_files = list(labels_dir.glob("*.nii.gz"))

        assert len(image_files) > 0, "No training images found"
        assert len(label_files) > 0, "No training labels found"

        # Check each image has corresponding label
        for img_file in image_files:
            # Remove _0000 suffix from image name to match label
            # img_file.stem removes .nii.gz, then remove .nii and _0000
            base_name = img_file.name.replace('_0000.nii.gz', '')
            label_file = labels_dir / f"{base_name}.nii.gz"
            assert label_file.exists(), f"Label file {label_file} missing for image {img_file}"

    def test_test_images_exist(self, setup_nnunet_paths, dataset_id):
        """Test test images exist."""
        raw_path = Path(os.environ["nnUNet_raw"])
        dataset_path = raw_path / f"Dataset{dataset_id:03d}_Hippocampus"

        test_dir = dataset_path / "imagesTs"
        test_files = list(test_dir.glob("*.nii.gz"))

        assert len(test_files) > 0, "No test images found"