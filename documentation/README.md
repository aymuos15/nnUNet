# nnU-Net Documentation

Complete documentation for nnU-Net v2.

## Getting Started

New to nnU-Net? Start here:

- **[Getting Started](getting_started.md)** - Installation, dataset preparation, and your first training run
- **[Core Concepts](core_concepts.md)** - How nnU-Net works: self-configuration, dataset fingerprints, and pipeline stages
- **[Advanced Usage](advanced_usage.md)** - Custom configurations, manual plan editing, and extending nnU-Net

## Reference Documentation

Detailed guides for specific topics:

### Dataset & Preprocessing
- **[Dataset Format](reference/dataset_format.md)** - How to structure your dataset for nnU-Net
- **[Intensity Normalization](reference/normalization.md)** - How nnU-Net normalizes image intensities
- **[Plans File](reference/plans_file.md)** - Understanding and editing nnUNetPlans.json

### Training
- **[Manual Data Splits](reference/manual_data_splits.md)** - Custom cross-validation splits
- **[Sparse Annotations](reference/ignore_label.md)** - Training with incomplete labels
- **[Region-Based Training](reference/region_based_training.md)** - Training on specific regions
- **[Pretraining & Fine-Tuning](reference/pretraining_and_finetuning.md)** - Transfer learning workflows

### Inference & Evaluation
- **[Pretrained Models](reference/pretrained_models.md)** - Using pretrained models for inference
- **[Inference Format](reference/inference_format.md)** - Dataset format for prediction-only

### Advanced Topics
- **[Extending nnU-Net](reference/extending_nnunet.md)** - Adding custom components
- **[Environment Variables](reference/environment_variables.md)** - Configuration via environment variables
- **[Residual Encoder Presets](reference/resenc_presets.md)** - High-performance ResEncUNet configurations

## Custom Architectures & Losses

- **[Custom Architecture Guide](../CUSTOM_ARCH_INFO.md)** - KiU-Net, UIU-Net, Blob Dice, and Region Dice integration

## Quick Links

### Installation & Setup
```bash
pip install nnunetv2
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Basic Workflow
```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train
nnUNetv2_train DATASET_ID 3d_fullres 0 --npz

# Predict
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 3d_fullres -f 0
```

### Command Reference

**Planning & Preprocessing**:
- `nnUNetv2_plan_and_preprocess` - Full pipeline
- `nnUNetv2_extract_fingerprint` - Extract dataset fingerprint only
- `nnUNetv2_plan_experiment` - Generate plans only
- `nnUNetv2_preprocess` - Preprocess only

**Training**:
- `nnUNetv2_train` - Train models

**Inference**:
- `nnUNetv2_predict` - Run inference
- `nnUNetv2_predict_from_modelfolder` - Predict using a model folder directly

**Evaluation**:
- `nnUNetv2_evaluate_folder` - Compute metrics on predictions
- `nnUNetv2_evaluate_simple` - Simple evaluation (Dice only)
- `nnUNetv2_find_best_configuration` - Compare configurations
- `nnUNetv2_accumulate_crossval_results` - Aggregate cross-validation results

**Postprocessing**:
- `nnUNetv2_determine_postprocessing` - Determine optimal postprocessing
- `nnUNetv2_apply_postprocessing` - Apply postprocessing to predictions

**Ensembling**:
- `nnUNetv2_ensemble` - Ensemble multiple models/configurations

**Utilities**:
- `nnUNetv2_plot_overlay_pngs` - Visualize predictions overlaid on images
- `nnUNetv2_convert_old_nnUNet_dataset` - Convert nnU-Net v1 datasets to v2 format
- `nnUNetv2_convert_MSD_dataset` - Convert Medical Segmentation Decathlon datasets

**Model Sharing**:
- `nnUNetv2_download_pretrained_model_by_url` - Download pretrained models
- `nnUNetv2_install_pretrained_model_from_zip` - Install pretrained model from ZIP
- `nnUNetv2_export_model_to_zip` - Export trained model to ZIP

**Advanced**:
- `nnUNetv2_move_plans_between_datasets` - Transfer plans for pretraining/fine-tuning

## Architecture

nnU-Net's modular architecture:

```
nnunetv2/
├── experiment_planning/   # Dataset analysis & plan generation
├── preprocessing/         # Data preprocessing
├── training/              # Training loop, losses, configs
├── architecture/          # Network architectures
├── inference/             # Prediction pipeline
├── postprocessing/        # Postprocessing strategies
├── evaluation/            # Metrics & result analysis
├── data/                  # Datasets & data loading
├── utilities/             # Helper functions
└── imageio/               # Image file I/O
```

Each component is designed to be modular and extensible.

## Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/MIC-DKFZ/nnUNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MIC-DKFZ/nnUNet/discussions)
- **Paper**: [Isensee et al., Nature Methods 2021](https://www.nature.com/articles/s41592-020-01008-z)

## Citation

```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
```
