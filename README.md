# nnU-Net

**Automatic medical image segmentation that adapts to your data.**

nnU-Net is a semantic segmentation framework that automatically configures itself for any medical imaging dataset. It analyzes your training data, designs an optimal U-Net architecture, and handles all preprocessing, training, and postprocessing without manual tuning.

## What is nnU-Net?

Medical image datasets vary enormously: 2D vs 3D, different modalities (CT, MRI, microscopy), varying resolutions, class imbalances, and target structures. Traditionally, each new dataset requires manual pipeline design and optimization—a process that's error-prone, time-consuming, and heavily dependent on expertise.

nnU-Net eliminates this burden. It automatically:
- Analyzes your dataset and creates a "fingerprint"
- Configures preprocessing (resampling, normalization)
- Designs network architecture (topology, patch size, batch size)
- Optimizes data augmentation
- Handles training and postprocessing

**No expertise required. Just provide your data.**

## Quickstart

### Installation

```bash
pip install nnunetv2
```

Or from source:
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

### Basic Usage

1. **Set up paths** (one-time setup):
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

2. **Prepare your dataset** in the nnU-Net format:
```
nnUNet_raw/
  Dataset001_MyDataset/
    imagesTr/          # Training images
    labelsTr/          # Training labels
    imagesTs/          # Test images (optional)
    dataset.json       # Dataset metadata
```

3. **Plan and preprocess**:
```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

4. **Train**:
```bash
nnUNetv2_train 001 3d_fullres 0
```

5. **Predict**:
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 001 -c 3d_fullres -f 0
```

## Key Features

- **Fully automatic**: Zero hyperparameter tuning required
- **Self-configuring**: Adapts to 2D/3D, any modality, any resolution
- **Battle-tested**: Used by winning solutions in numerous medical imaging challenges
- **Extensible**: Framework for developing new segmentation methods
- **Custom architectures**: Supports KiU-Net, UIU-Net, and other custom networks
- **Instance-aware losses**: Blob Dice and Region Dice for specialized segmentation tasks

## Documentation

- **[Getting Started](documentation/getting_started.md)** - Installation, dataset preparation, first training run
- **[Core Concepts](documentation/core_concepts.md)** - How nnU-Net works under the hood
- **[Advanced Usage](documentation/advanced_usage.md)** - Fine-tuning, custom configurations, extending nnU-Net
- **[Custom Architectures](CUSTOM_ARCH_INFO.md)** - KiU-Net, UIU-Net integration guide

For more details, see the [documentation/](documentation/) folder.

## Who Should Use nnU-Net?

**Domain Scientists** (biologists, radiologists): Get state-of-the-art segmentation on your images without deep learning expertise.

**AI Researchers**: Use nnU-Net as a strong baseline, method development framework, or starting point for challenge submissions.

## Citation

If you use nnU-Net, please cite:

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

## License

Apache License 2.0 - See [LICENSE](LICENSE)

## Acknowledgements

Developed by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) and the Division of Medical Image Computing at the [German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
