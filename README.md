# Oral Cancer Prediction using Deep Learning

A medical image classification system that leverages transfer learning and convolutional neural networks to detect oral cancer from clinical images of the oral cavity.

## Abstract

This project implements a binary classification model to distinguish between cancerous and non-cancerous oral cavity images. The system utilizes the VGG16 architecture pre-trained on ImageNet, combined with custom classification layers, to achieve accurate predictions. This work aims to support early detection of oral cancer through automated image analysis.

## Table of Contents

- [Introduction](#introduction)
- [Technical Approach](#technical-approach)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results and Discussion](#results-and-discussion)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Introduction

Oral cancer is a significant health concern worldwide, with early detection being critical for successful treatment outcomes. This project develops an automated classification system using deep learning techniques to assist in the screening process. The model analyzes images of the oral cavity and provides binary classification results indicating the presence or absence of cancerous lesions.

### Objectives

- Develop a robust binary classification model for oral cancer detection
- Implement transfer learning to overcome limited dataset constraints
- Achieve clinically relevant accuracy metrics
- Create a reproducible and extensible codebase for further research

## Technical Approach

### Transfer Learning

The project employs transfer learning using the VGG16 architecture, originally trained on the ImageNet dataset. This approach leverages pre-trained feature extractors to compensate for limited medical imaging data, a common challenge in healthcare machine learning applications.

### Two-Phase Training Strategy

**Phase 1: Feature Extraction**
- Pre-trained VGG16 layers remain frozen
- Custom classification head is trained
- Duration: 10 epochs
- Learning rate: 0.0001

**Phase 2: Fine-Tuning**
- All model layers become trainable
- End-to-end fine-tuning on domain-specific data
- Duration: 5 epochs
- Learning rate: 0.00001

## System Architecture

### Model Components
Input Layer (128x128x3)
        ↓
VGG16 Base Model (Pre-trained)
        ↓
Flatten Layer
        ↓
Dense Layer (256 units, ReLU activation)
        ↓
Dropout Layer (rate=0.5)
        ↓
Output Layer (1 unit, Sigmoid activation)

### Network Specifications

- **Input Shape**: 128x128x3 (RGB images)
- **Base Architecture**: VGG16 (without top classification layers)
- **Activation Functions**: ReLU (hidden layers), Sigmoid (output layer)
- **Regularization**: Dropout (0.5)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

## Dataset

### Data Sources

The training dataset comprises medical images sourced from publicly available Kaggle datasets focused on oral cancer detection.

### Dataset Composition

- **Total Images**: 106
- **Training Set**: 85 images (80%)
- **Validation Set**: 21 images (20%)
- **Classes**: 2 (Cancer, Non-Cancer)
- **Image Format**: JPEG/PNG
- **Image Dimensions**: 128x128 pixels (after preprocessing)

### Data Preprocessing

- Rescaling: Pixel values normalized to [0, 1] range
- Data augmentation applied during training:
  - Rotation range: 20 degrees
  - Width shift: 0.2
  - Height shift: 0.2
  - Shear transformation: 0.2
  - Zoom range: 0.2
  - Horizontal flip:dataset/

### Directory Structure
├── cancer/
│ ├── image001.jpg
│ ├── image002.jpg
│ └── ...
└── non-cancer/
├── image001.jpg
├── image002.jpg
└── ... 

## Model Performance

### Training Metrics

| Metric | Phase 1 | Phase 2 (Fine-tuned) |
|--------|---------|----------------------|
| Training Accuracy | 75-80% | 88% |
| Validation Accuracy | 84% | 84-88% |
| Training Loss | 0.50 | 0.27 |
| Validation Loss | 0.44 | 0.37 |

### Evaluation

The model demonstrates consistent performance across training and validation sets, with validation accuracy reaching approximately 84-88%. The relatively small gap between training and validation metrics suggests adequate generalization despite the limited dataset size.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Dependencies

pip install tensorflow>=2.12.0
pip install numpy>=1.23.0
pip install matplotlib>=3.7.0
pip install Pillow>=9.5.0
pip install pandas>=2.0.0


### Setup Instructions

Clone the repository
git clone https://github.com/Harsha2oo5/Oral_cancer.git

Navigate to project directory
cd Oral_cancer

Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


## Usage

### Training the Model

Import required libraries
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Configure data generators
train_datagen = ImageDataGenerator(
rescale=1./255,
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
validation_split=0.2
)

Load and train model
See oral-cancer-prediction.ipynb for complete implementation


### Inference on New Images

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

Load trained model
model = load_model('oral_cancer_vgg16_model.h5')

Load and preprocess image
img_path = 'path/to/test/image.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

Generate prediction
prediction_score = model.predict(img_array)

Interpret results
if prediction_score > 0.5:
print(f"Classification: Non-Cancer (Confidence: {prediction_score:.4f})")
else:
print(f"Classification: Cancer (Confidence: {1-prediction_score:.4f})")


### Batch Prediction

import os

Directory containing test images
test_directory = 'path/to/test/images'

Process all images in directory
for filename in os.listdir(test_directory):
if filename.endswith(('.jpg', '.jpeg', '.png')):
img_path = os.path.join(test_directory, filename)
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

  prediction = model.predict(img_array)
  result = 'Non-Cancer' if prediction > 0.5 else 'Cancer'
  print(f"{filename}: {result} ({prediction:.4f})")


## Project Structure

Oral_cancer/
│
├── oral-cancer-prediction.ipynb # Main Jupyter notebook
├── oral_cancer_prediction_dataset.csv # Dataset metadata
├── README.md # Project documentation
├── LICENSE # License information
├── requirements.txt # Python dependencies
│
├── models/ # Saved model files
│ └── oral_cancer_vgg16_model.h5
│
├── data/ # Dataset directory
│ ├── cancer/
│ └── non-cancer/
│
└── outputs/ # Generated outputs
├── training_plots/
└── predictions/


## Requirements

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training possible
- **Recommended**: 16GB RAM, NVIDIA GPU with CUDA support
- **Storage**: 5GB free space

### Software Requirements

- Python 3.8+
- TensorFlow 2.12+
- CUDA 11.8 and cuDNN 8.6 (for GPU acceleration)
- Jupyter Notebook or JupyterLab

### Python Package Dependencies

tensorflow>=2.12.0
numpy>=1.23.0
matplotlib>=3.7.0
Pillow>=9.5.0
pandas>=2.0.0
scikit-learn>=1.3.0


## Results and Discussion

### Key Findings

The implemented model achieves validation accuracy of 84-88%, demonstrating the effectiveness of transfer learning for medical image classification with limited training data. The VGG16 architecture, despite being trained on general-purpose images, successfully transfers learned features to the medical imaging domain.

### Performance Analysis

- **Strengths**: High accuracy on validation set, stable training curve, effective use of data augmentation
- **Observations**: Minimal overfitting due to dropout and regularization techniques
- **Validation**: Consistent performance across multiple training runs

## Limitations

### Dataset Constraints

- Limited dataset size (106 images) may restrict model generalization
- Class imbalance not thoroughly addressed
- Lack of diverse demographic representation in training data

### Model Constraints

- Binary classification only (does not distinguish between cancer types)
- No uncertainty quantification in predictions
- Performance on edge cases not extensively evaluated

### Clinical Deployment Considerations

- Model requires validation on independent clinical datasets
- Not designed to replace professional medical diagnosis
- Interpretability features (e.g., attention maps) not implemented

## Future Work

### Short-term Improvements

- Expand dataset to 1000+ images per class
- Implement k-fold cross-validation for robust evaluation
- Add confusion matrix and ROC curve analysis
- Integrate Grad-CAM for model interpretability

### Long-term Extensions

- Multi-class classification for different oral pathologies
- Development of web-based deployment interface
- Integration with clinical workflows
- Exploration of ensemble methods combining multiple architectures
- Implementation of uncertainty estimation techniques

### Research Directions

- Investigation of attention mechanisms for improved feature learning
- Comparison with state-of-the-art architectures (EfficientNet, ResNet, Vision Transformers)
- Study of model robustness to image quality variations
- Development of explainable AI methods for clinical acceptance

## Contributing

Contributions to this project are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines for Python code
- Include docstrings for all functions and classes
- Add unit tests for new functionality
- Update documentation as needed

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

## References

1. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556.

2. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition.

3. Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10), 1345-1359.

4. Chollet, F. (2017). Deep Learning with Python. Manning Publications.

## Contact

**Project Maintainer**: K Sai Sri Harsha

**GitHub**: [https://github.com/Harsha2oo5](https://github.com/Harsha2oo5)

**Project Repository**: [https://github.com/Harsha2oo5/Oral_cancer](https://github.com/Harsha2oo5/Oral_cancer)

For questions, suggestions, or collaboration opportunities, please open an issue on the GitHub repository.

---

**Medical Disclaimer**: This software is intended for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice and diagnosis. The authors and contributors assume no liability for any clinical use of this system.

**Last Updated**: November 2025





