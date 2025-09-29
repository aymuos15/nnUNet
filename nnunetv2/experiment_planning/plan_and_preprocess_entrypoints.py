# Import new modular entry points
from nnunetv2.experiment_planning.cli.entrypoints import (
    extract_fingerprint_entry as _extract_fingerprint_entry,
    plan_experiment_entry as _plan_experiment_entry,
    preprocess_entry as _preprocess_entry,
    plan_and_preprocess_entry as _plan_and_preprocess_entry
)


def extract_fingerprint_entry():
    """Entry point for dataset fingerprint extraction (uses new modular structure)."""
    _extract_fingerprint_entry()


def plan_experiment_entry():
    """Entry point for experiment planning (uses new modular structure)."""
    _plan_experiment_entry()


def preprocess_entry():
    """Entry point for preprocessing (uses new modular structure)."""
    _preprocess_entry()


def plan_and_preprocess_entry():
    """Entry point for combined planning and preprocessing (uses new modular structure)."""
    _plan_and_preprocess_entry()


if __name__ == '__main__':
    plan_and_preprocess_entry()
