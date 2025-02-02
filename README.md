# Transfer Learning for Image Classification

## Overview
This project demonstrates the use of Transfer Learning to classify images into multiple categories. A pre-trained ResNet model was fine-tuned to achieve high accuracy on a custom dataset.

## Key Features
- Utilized **ResNet** architecture with pre-trained weights for feature extraction.
- Fine-tuned the model to optimize classification performance on a new dataset.
- Employed techniques like **Batch Normalization** and **Dropout** to prevent overfitting.

## Results
- Initial accuracy: **89%**
- Final accuracy after fine-tuning: **93%**

## Technologies and Tools
- **Frameworks**: PyTorch, TensorFlow
- **Libraries**: NumPy, Matplotlib, Scikit-learn
- **Techniques**: Batch Normalization, Dropout, Optimizer Tuning

## Challenges
- **Overfitting**: Mitigated using regularization techniques.
- **Training Time**: Managed by reducing the learning rate during fine-tuning.

## How to Run
1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib scikit-learn
