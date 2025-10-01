# nnunetv2 Package

This is the main package containing all nnU-Net v2 functionality. The package is organized into modular components, each handling a specific aspect of the automatic segmentation pipeline.

## Package Structure

### Core Pipeline Components

- **[experiment_planning/](experiment_planning/)** - Dataset analysis and pipeline configuration
  - Analyzes training data to create dataset "fingerprints"
  - Generates training plans (architecture, preprocessing, etc.)
  - Configures optimal settings based on dataset properties

- **[preprocessing/](preprocessing/)** - Data preprocessing operations
  - Resampling to target spacing
  - Intensity normalization
  - Image cropping and padding

- **[training/](training/)** - Model training infrastructure
  - Trainer classes orchestrating the training loop
  - Loss functions (Dice, Cross-Entropy, Blob Dice, Region Dice)
  - Learning rate schedulers
  - Configuration system for custom architectures

- **[inference/](inference/)** - Prediction pipeline
  - Sliding window prediction for large images
  - Test-time augmentation (mirroring)
  - Multi-model ensembling
  - Efficient memory management

- **[postprocessing/](postprocessing/)** - Post-prediction refinement
  - Connected component analysis
  - Automatic postprocessing determination

### Supporting Components

- **[architecture/](architecture/)** - Network architectures
  - Dynamic U-Net builder
  - Custom architectures (KiU-Net, UIU-Net)
  - Deep supervision handling

- **[data/](data/)** - Dataset and data loading
  - Dataset abstractions
  - Data loaders with augmentation
  - Transforms and augmentation strategies

- **[evaluation/](evaluation/)** - Results analysis
  - Cross-validation aggregation
  - Configuration comparison
  - Metric computation (Dice, Hausdorff distance, etc.)

- **[utilities/](utilities/)** - Helper functions
  - Visualization tools
  - File operations
  - Class discovery and dynamic imports

- **[imageio/](imageio/)** - Image file I/O
  - Multi-format support (NIfTI, TIFF, PNG, etc.)
  - Pluggable reader/writer system

- **[model_sharing/](model_sharing/)** - Pretrained model management
  - Model export/import
  - Pretrained model download
  - ZIP-based model distribution

## How It Works

### 1. Experiment Planning (`experiment_planning/`)

When you run `nnUNetv2_plan_and_preprocess`, nnU-Net:
1. Extracts a dataset fingerprint (image sizes, spacings, intensities, class distribution)
2. Creates multiple configurations (2D, 3D full-res, 3D low-res → cascade)
3. Determines optimal preprocessing, network topology, patch size, and batch size
4. Saves plans to disk

### 2. Training (`training/`)

When you run `nnUNetv2_train`, nnU-Net:
1. Loads the training plan
2. Applies preprocessing on-the-fly
3. Builds the network architecture
4. Trains with automatic data augmentation
5. Validates periodically and saves checkpoints

### 3. Inference (`inference/`)

When you run `nnUNetv2_predict`, nnU-Net:
1. Loads the trained model
2. Applies preprocessing to input images
3. Performs sliding window prediction
4. Applies postprocessing
5. Exports predictions

## Key Design Principles

### Three-Step Configuration Recipe

1. **Fixed parameters**: Robust defaults that work universally (loss function, most augmentation, learning rate)
2. **Rule-based parameters**: Hard-coded heuristics adapted from dataset fingerprint (network topology, patch size, batch size)
3. **Empirical parameters**: Trial-and-error (best configuration selection, postprocessing strategy)

### Modularity

The codebase is organized into small, focused modules rather than monolithic classes. This makes it easier to:
- Understand individual components
- Extend functionality
- Reuse code across different contexts
- Debug and test

### Extensibility

nnU-Net is designed to be a framework, not just a tool:
- Custom network architectures via `TrainerConfig`
- Custom loss functions
- Custom preprocessing strategies
- Custom planning heuristics

## For Developers

### Adding Custom Components

- **Custom architecture**: Implement in `architecture/custom/`, register in `training/configs/`
- **Custom loss**: Implement in `training/losses/implementations/`, use in `TrainerConfig`
- **Custom planner**: Subclass `ExperimentPlanner` in `experiment_planning/planners/`

See `CUSTOM_ARCH_INFO.md` at the repository root for examples.

### Code Organization

Each major component follows a similar structure:
```
component/
  ├── __init__.py          # Public API
  ├── base/                # Abstract base classes
  ├── implementations/     # Concrete implementations
  ├── utils/               # Helper functions
  └── config/              # Configuration objects
```

## CLI Entry Points

The package exposes several command-line tools (defined in `pyproject.toml`):

- `nnUNetv2_plan_and_preprocess` - Full pipeline: fingerprint → plan → preprocess
- `nnUNetv2_train` - Train models
- `nnUNetv2_predict` - Run inference
- `nnUNetv2_find_best_configuration` - Compare 2D/3D/cascade results
- `nnUNetv2_ensemble` - Ensemble multiple models
- `nnUNetv2_evaluate_folder` - Compute metrics

See `--help` on any command for details.

## Environment Variables

nnU-Net uses environment variables for path configuration:

- `nnUNet_raw` - Raw dataset storage
- `nnUNet_preprocessed` - Preprocessed data storage  
- `nnUNet_results` - Model outputs and checkpoints
- `nnUNet_compile` - Enable/disable torch.compile (default: true)

See the root `README.md` for setup instructions.
