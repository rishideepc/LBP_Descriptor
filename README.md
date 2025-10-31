# Texture and Medical Image Classification using Local Binary Patterns (LBP)

## Overview

This project implements a texture and medical image classification system using **Local Binary Pattern (LBP)** descriptors combined with machine learning classifiers. The system is designed to classify both texture patterns from the Brodatz dataset and brain MRI scans from the OASIS medical database.

This work was completed as part of an undergraduate thesis, exploring the effectiveness of LBP-based feature extraction for multi-class image classification tasks.

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [Requirements](#requirements)

## Features

- **LBP Feature Extraction**: Implements uniform Local Binary Pattern feature extraction with configurable radius and neighboring points
- **Region-Based Thresholding**: Divides images into non-overlapping regions for localized histogram thresholding
- **Multiple Classifiers**: Supports various machine learning models including:
  - Support Vector Machines (SVM)
  - Random Forest
  - Multi-Layer Perceptron (MLP)
  - K-Nearest Neighbors (KNN)
  - One-vs-Rest Classifier with SVM
- **Dual Dataset Support**: Works with both texture patterns (Brodatz) and medical images (OASIS MRI)
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and F1-score metrics
- **Image Preprocessing**: Includes grayscale conversion, filtering, and normalization
- **GIF to JPG Conversion**: Utility for converting GIF medical images to JPG format

## Datasets

### 1. Brodatz Texture Database
- Contains various texture patterns (wood, fabric, natural textures, etc.)
- Used for binary classification: **regular** vs **irregular** textures
- Images stored in: `assets/textures/`
- Labels defined in: `Labels.xlsx`

### 2. OASIS MRI Database
- **OASIS**: Open Access Series of Imaging Studies
- Contains brain MRI scans organized into 4 categories
- Dataset structure:
  - `OASIS_Cross_1/` and `OASIS_Cross_1_converted/`
  - `OASIS_Cross_2/` and `OASIS_Cross_2_converted/`
  - `OASIS_Cross_3/` and `OASIS_Cross_3_converted/`
  - `OASIS_Cross_4/` and `OASIS_Cross_4_converted/`
- Original GIF images converted to JPG format for processing
- Multi-class classification task (4 categories)

## Methodology

### 1. Image Preprocessing
- Load images in grayscale format
- Optional median blur filtering (kernel size: 5x5) for noise reduction
- Image resizing to standardized dimensions (64x64 pixels)

### 2. LBP Feature Extraction
- **Parameters**:
  - Radius: 1
  - Number of points: 8 (8 * radius)
  - Method: uniform LBP
- Extract LBP feature map from grayscale images
- Normalize LBP values to range [0, 255]

### 3. Region-Based Histogram Thresholding
- Divide LBP image into non-overlapping regions (20x20 pixels)
- Compute histogram for each region (9 bins)
- Apply adaptive thresholding:
  - Normalize histogram
  - Calculate threshold as: `threshold_factor * mean(histogram)`
  - Binarize histogram values
- Concatenate regional histograms to form final feature vector

### 4. Classification
- Split data: 80% training, 20% testing (OASIS) or 85% training, 15% testing (Brodatz)
- Train classifier on extracted features
- Evaluate on test set using multiple metrics

## Project Structure

```
LBP_Descriptor/
│
├── main.py                 # Main implementation for OASIS MRI classification
├── oasis_db.py            # OASIS dataset specific implementation
├── brodatz_db.py          # Brodatz texture dataset implementation
├── converter.py           # GIF to JPG conversion utility
├── Labels.xlsx            # Texture classification labels
│
└── assets/
    ├── textures/          # Brodatz texture database images (.tiff)
    ├── OASIS_MRI_DB/      # OASIS MRI brain scans
    │   ├── OASIS_Cross_1/ to OASIS_Cross_4/
    │   └── OASIS_Cross_1_converted/ to OASIS_Cross_4_converted/
    ├── face/              # Face image samples
    └── paper_examples/    # Example images
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install numpy
pip install pandas
pip install imageio
pip install openpyxl
```

Or install all dependencies at once:

```bash
pip install opencv-python scikit-image scikit-learn numpy pandas imageio openpyxl
```

## Usage

### Running OASIS MRI Classification

```bash
python main.py
```

or

```bash
python oasis_db.py
```

**Configuration parameters** (in the script):
- `radius`: LBP radius (default: 1)
- `n_points`: Number of neighboring points (default: 8)
- `region_size`: Size of non-overlapping regions (default: 20)
- `threshold_factor`: Threshold multiplier (default: 0.3 for OASIS)

### Running Brodatz Texture Classification

```bash
python brodatz_db.py
```

**Configuration parameters**:
- Same LBP parameters as above
- `threshold_factor`: Default 0.3 for texture classification
- Uses `Labels.xlsx` for texture labels

### Converting GIF to JPG (for OASIS dataset)

```bash
python converter.py
```

**Note**: Update the directory paths in the script before running.

## Results

The system evaluates classification performance using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision across classes
- **Recall**: Weighted average recall across classes
- **F1-Score**: Weighted average F1-score across classes

### Example Output:

```
Accuracy on the test set: 85.23%
Precision: 84.56 %
Recall: 85.23 %
F1 Score: 84.78 %
```

## Technical Details

### LBP Feature Extraction

Local Binary Patterns is a texture descriptor that labels pixels by thresholding the neighborhood of each pixel and considers the result as a binary number.

**Algorithm**:
1. For each pixel, compare it with its 8 neighbors
2. If neighbor value ≥ center pixel, assign 1; else assign 0
3. Concatenate binary values to form 8-bit binary number
4. Convert to decimal (0-255) - this is the LBP code

### Region-Based Thresholding

Instead of using raw LBP histograms, this implementation:
1. Divides the LBP image into regions
2. Computes localized histograms
3. Applies adaptive thresholding to each region
4. Creates a more discriminative feature representation

### Helper Functions

- `divide_into_regions(lbp, region_size)`: Divides LBP image into non-overlapping regions
- `normalize_histogram(histogram)`: Normalizes histogram to sum to 1
- `threshold_histogram(histogram, threshold_factor)`: Applies binary thresholding to histogram
- `normalize_lbp_image(lbp_image)`: Normalizes LBP feature map to [0, 255] range
- `extract_features(image_path, threshold_factor)`: Main feature extraction pipeline

## Requirements

```
opencv-python>=4.5.0
scikit-image>=0.18.0
scikit-learn>=0.24.0
numpy>=1.19.0
pandas>=1.2.0
imageio>=2.9.0
openpyxl>=3.0.0
```

## Machine Learning Models Tested

The project experiments with multiple classifiers:

1. **Support Vector Machine (SVM)**
   - Linear kernel
   - RBF kernel (default in OneVsRestClassifier)

2. **Random Forest Classifier**
   - n_estimators: 100
   - Ensemble of decision trees

3. **Multi-Layer Perceptron (MLP)**
   - Hidden layers: (100, 100)
   - max_iter: 1000

4. **K-Nearest Neighbors (KNN)**
   - n_neighbors: 3

5. **One-vs-Rest SVM** (Primary model)
   - Multi-class classification strategy
   - SVM with default parameters

## Data Flow

```
Input Image
    ↓
Grayscale Conversion
    ↓
Optional Filtering
    ↓
LBP Feature Extraction
    ↓
Region Division
    ↓
Histogram Computation per Region
    ↓
Thresholding & Normalization
    ↓
Feature Vector Concatenation
    ↓
Classification
    ↓
Performance Metrics
```

## Future Improvements

Potential enhancements for this project:

- Implement cross-validation for more robust evaluation
- Add data augmentation for improved generalization
- Experiment with multi-scale LBP (different radius values)
- Include rotation-invariant LBP variants
- Add visualization of LBP feature maps
- Implement grid search for hyperparameter optimization
- Add confusion matrix visualization
- Support for additional medical imaging modalities

## References

- T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2002.
- OASIS Brains Dataset: https://www.oasis-brains.org/
- Brodatz Texture Database: http://www.ux.uis.no/~tranden/brodatz.html

## Author

Undergraduate Thesis Project

## License

This project is part of an academic thesis and is intended for educational and research purposes.

---

**Last Updated**: 2025

For questions or issues, please refer to the code comments or contact the author.
