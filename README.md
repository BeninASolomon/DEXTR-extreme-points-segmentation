# DEXTR Extreme Points Segmentation

This repository contains the implementation of **DEXTR (Deep Extreme Cut)**, a model designed for image segmentation using extreme points as guidance. The model combines a U-Net-like architecture with a custom `ExtremePointsLayer` to process extreme points for efficient segmentation.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Pretrained Model](#pretrained-model)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Extreme Points Integration**: This model takes in extreme points as input along with the image to enhance segmentation accuracy.
- **U-Net-like Architecture**: Utilizes an encoder-decoder structure with skip connections for effective feature extraction and image reconstruction.
- **Custom Layer**: A custom `ExtremePointsLayer` to tile and reshape extreme points into the network’s feature maps.
- **Supports Grayscale Images**: Model works with 100x100 grayscale images, but can be adjusted for different sizes.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib

## Installation
First, clone the repository:
```bash
git clone https://github.com/BeninASolomon/DEXTR-extreme-points-segmentation.git
cd DEXTR-extreme-points-segmentation
pip install tensorflow keras opencv-python numpy matplotlib
