import os
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def setup_nnunet_paths():
    """Setup nnUNet environment variables for testing."""
    original_env = {}

    # Store original values
    for key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        original_env[key] = os.environ.get(key)

    # Set test paths
    base_path = Path("/home/localssk23/nnn/datasets")
    os.environ["nnUNet_raw"] = str(base_path / "raw")
    os.environ["nnUNet_preprocessed"] = str(base_path / "preprocessed")
    os.environ["nnUNet_results"] = str(base_path / "results")

    # Create directories
    os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)
    os.makedirs(os.environ["nnUNet_results"], exist_ok=True)

    yield

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def dataset_id():
    """Dataset ID for testing."""
    return 4