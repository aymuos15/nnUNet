# Image I/O

This module provides a pluggable system for reading and writing medical image files in various formats.

## Overview

The imageio module handles:

- **Multi-format support** - NIfTI, TIFF, PNG, JPEG, BMP, NPY, NPZ
- **Automatic format detection** - Based on file extension
- **Reader/Writer registry** - Extensible system for custom formats
- **Metadata preservation** - Spacing, orientation, etc.

## Directory Structure

```
imageio/
├── base_reader_writer.py           # Abstract base class
├── reader_writer_registry.py       # Format registry
├── nibabel_reader_writer.py        # NIfTI support (via nibabel)
├── simpleitk_reader_writer.py      # NIfTI/NRRD support (via SimpleITK)
├── tif_reader_writer.py            # TIFF support
├── natural_image_reader_writer.py  # PNG/JPEG/BMP support
└── __init__.py
```

## Supported Formats

| Format | Extension | Reader | Writer | Metadata Support |
|--------|-----------|--------|--------|------------------|
| NIfTI | `.nii`, `.nii.gz` | ✓ | ✓ | Spacing, orientation |
| NRRD | `.nrrd`, `.nhdr` | ✓ | ✓ | Spacing, orientation |
| TIFF | `.tif`, `.tiff` | ✓ | ✓ | Spacing (via metadata) |
| PNG | `.png` | ✓ | ✓ | No metadata |
| JPEG | `.jpg`, `.jpeg` | ✓ | ✓ | No metadata |
| BMP | `.bmp` | ✓ | ✓ | No metadata |
| NumPy | `.npy`, `.npz` | ✓ | ✓ | No metadata |

## Usage

### Reading Images

**Automatic format detection**:

```python
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

# Determine reader from dataset.json
rw = determine_reader_writer_from_dataset_json(
    dataset_json,
    example_file='case_001_0000.nii.gz'
)

# Read image
image, properties = rw.read_images(['case_001_0000.nii.gz'])
# image: numpy array, shape [C, X, Y, Z] or [C, X, Y]
# properties: dict with 'spacing', 'original_shape', etc.
```

**Manual format selection**:

```python
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# Read NIfTI with SimpleITK
rw = SimpleITKIO()
image, properties = rw.read_images(['image.nii.gz'])

# Read segmentation
seg, props = rw.read_seg('segmentation.nii.gz')
```

### Writing Images

```python
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

rw = SimpleITKIO()

# Write image
rw.write_seg(
    seg=segmentation_array,
    output_fname='output.nii.gz',
    properties={'spacing': [1.0, 1.0, 1.0], 'origin': [0, 0, 0]}
)
```

### Multi-Channel Images

Reading multi-channel (multi-modal) images:

```python
# Read multiple files as channels
image, properties = rw.read_images([
    'case_001_0000.nii.gz',  # Channel 0
    'case_001_0001.nii.gz',  # Channel 1
    'case_001_0002.nii.gz',  # Channel 2
])

# image.shape: [3, X, Y, Z]
```

## Reader/Writer Classes

### NibabelIO (`nibabel_reader_writer.py`)

Uses `nibabel` library for NIfTI files:

- **Pros**: Pure Python, fast
- **Cons**: Limited format support (mainly NIfTI)

```python
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO

rw = NibabelIO()
image, props = rw.read_images(['image.nii.gz'])
```

### SimpleITKIO (`simpleitk_reader_writer.py`)

Uses `SimpleITK` library for medical images:

- **Pros**: Supports many formats (NIfTI, NRRD, DICOM series, etc.)
- **Cons**: Requires compiled library

```python
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

rw = SimpleITKIO()
image, props = rw.read_images(['image.nrrd'])
```

### TiffIO (`tif_reader_writer.py`)

Uses `tifffile` for TIFF images:

- **Supports**: Multi-page TIFF, 2D/3D
- **Metadata**: Resolution tags (spacing)

```python
from nnunetv2.imageio.tif_reader_writer import TiffIO

rw = TiffIO()
image, props = rw.read_images(['image.tif'])
```

### NaturalImageIO (`natural_image_reader_writer.py`)

Uses `PIL/Pillow` for natural images:

- **Supports**: PNG, JPEG, BMP
- **Use case**: 2D natural images, microscopy

```python
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO

rw = NaturalImage2DIO()
image, props = rw.read_images(['image.png'])
```

## Custom Reader/Writer

To support a new format:

### 1. Implement Reader/Writer

```python
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import numpy as np

class MyFormatIO(BaseReaderWriter):
    supported_file_endings = ['.myformat']
    
    def read_images(self, image_fnames):
        """Read image files."""
        # Load image data
        data = my_load_function(image_fnames[0])
        
        # Convert to numpy array [C, X, Y, Z]
        image = np.array(data)[None]  # Add channel dimension
        
        # Extract properties
        properties = {
            'spacing': [1.0, 1.0, 1.0],
            'origin': [0, 0, 0],
            'direction': np.eye(3)
        }
        
        return image, properties
    
    def read_seg(self, seg_fname):
        """Read segmentation file."""
        seg = my_load_function(seg_fname)
        return seg, properties
    
    def write_seg(self, seg, output_fname, properties):
        """Write segmentation file."""
        my_save_function(seg, output_fname, properties)
```

### 2. Register Reader/Writer

```python
from nnunetv2.imageio.reader_writer_registry import register_reader_writer

register_reader_writer(MyFormatIO, '.myformat')
```

### 3. Use in Dataset

Update `dataset.json`:

```json
{
  "file_ending": ".myformat",
  ...
}
```

nnU-Net will automatically use your custom reader/writer.

## Properties Dictionary

Reader functions return a `properties` dict with metadata:

**Required keys**:
- `spacing` - Voxel spacing, list of floats `[x, y, z]` or `[x, y]`

**Optional keys**:
- `origin` - Image origin in world coordinates
- `direction` - Orientation matrix
- `original_shape` - Shape before any preprocessing
- `original_spacing` - Spacing before resampling
- `crop_bbox` - Bounding box if cropped `[[x_min, x_max], [y_min, y_max], ...]`

**Used for**:
- Resampling images to target spacing
- Restoring predictions to original space
- Ensuring spatial consistency

## Tips

### Format Selection

**NIfTI (medical 3D)**:
- Use `SimpleITKIO` or `NibabelIO`
- Standard format for medical imaging

**TIFF (microscopy)**:
- Use `TiffIO`
- Supports multi-page for 3D

**PNG/JPEG (2D natural)**:
- Use `NaturalImage2DIO`
- For histopathology, microscopy 2D

### Performance

**Reading speed**:
- NibabelIO is fastest for NIfTI
- SimpleITKIO is more versatile but slightly slower

**Writing speed**:
- Compressed formats (`.nii.gz`) are slower to write
- Use uncompressed (`.nii`) for faster I/O, then compress later

### Memory

**Lazy loading**:
- Readers load entire file into memory
- For very large images, consider memory-mapped arrays

### Debugging

**Check properties**:
```python
image, props = rw.read_images(['image.nii.gz'])
print(f"Shape: {image.shape}")
print(f"Spacing: {props['spacing']}")
print(f"Origin: {props.get('origin', 'Not available')}")
```

**Validate format**:
```python
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_file_ending

rw = determine_reader_writer_from_file_ending('.nii.gz')
print(f"Reader: {rw.__class__.__name__}")
```

## See Also

- [Dataset Format Reference](../documentation/reference/dataset_format.md) - How to structure datasets
- [Preprocessing](../preprocessing/) - How imageio is used in preprocessing
- [Inference](../inference/) - How imageio is used in prediction
