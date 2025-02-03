# Transfer Learning for Image Classification using ResNet

## Overview
This project demonstrates the use of **Transfer Learning** to classify images from the **FashionMNIST dataset**. A pre-trained **ResNet** model was fine-tuned to achieve high accuracy on this dataset, which consists of grayscale images of fashion items belonging to 10 different categories.

## Key Features
- Utilized **ResNet architecture** with pre-trained weights for feature extraction.
- Fine-tuned the model to optimize classification performance on **FashionMNIST**.
- Applied techniques such as **Batch Normalization** and **Dropout** to reduce overfitting.
- Used **learning rate scheduling** to improve convergence and stability during training.

## Dataset: FashionMNIST
- **Source**: [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Description**: 70,000 grayscale images of clothing items (28x28 pixels).
- **Classes**: 10 categories, including T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Results
- **Initial accuracy** (using frozen layers): **88%**
- **Final accuracy** (after fine-tuning): **94%**

## Technologies and Tools
- **Frameworks**: PyTorch, TensorFlow
- **Libraries**: NumPy, Matplotlib, Scikit-learn
- **Techniques**: Transfer Learning, Batch Normalization, Dropout, Learning Rate Scheduling, Adam Optimizer

## Challenges
- **Overfitting**: Mitigated using **Dropout**, **Data Augmentation**, and **L2 Regularization**.
- **Training Time**: Reduced using **early stopping** and an **adaptive learning rate**.

## How to Run
1. Run the Jupyter Notebook (`ResNet_FashionMNIST.ipynb`) using Jupyter or Google Colab.
2. Train the model on FashionMNIST by executing all cells.
3. Evaluate the model on the test set to observe the classification performance.

## Further Details
You can find more detailed explanations in the notebook. Additionally, a comparison with a **CNN model** tested on the same dataset is included, along with some interesting conclusions.

## Future Improvements
- Implement **ResNet-101** to compare performance with **ResNet-50**.
- Experiment with **different optimizers** like **SGD with momentum**.
- Improve accuracy further using **ensemble learning**.

## References
- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **FashionMNIST Dataset**: [GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)
- **PyTorch ResNet Implementation**: [PyTorch Docs](https://pytorch.org/docs/stable/torchvision/models.html)


