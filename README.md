# EfficientViT-B4 Pretraining on ImageNet-1k

This repository contains the code and configuration for pretraining the **EfficientViT-B4** model on the **ImageNet-1k** dataset. The model is designed for efficient vision processing with optimized performance and resource utilization.

## Repository Overview

- `pre.py`: Main script to activate the pretraining of the EfficientViT-B4 model using the ImageNet-1k dataset.
- `checkpoints/`: Directory for saving model checkpoints during training (best and latest).
- `README.md`: Project description and usage guide.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/anto18671/efficientvit-b4.r256.git
cd efficientvit-b4.r256
pip install -r requirements.txt
```

The dependencies include:
- **PyTorch**
- **torchvision**
- **timm** (PyTorch Image Models)
- **Hugging Face `datasets`**
- **torchsummary**
- **tqdm**

## Dataset

The pretraining uses the **ImageNet-1k** dataset, which consists of 1.2 million images across 1000 categories. The dataset is automatically loaded using Hugging Face's `datasets` library.

## Pretraining

To start the pretraining process, simply run the `pre.py` script:

```bash
python pre.py
```

This script:
- Initializes the EfficientViT-B4 model.
- Sets up the data pipelines with transformations (resizing, augmentation, normalization).
- Configures the optimizer (AdamW) and the learning rate scheduler.
- Begins the pretraining from scratch or resumes from the last saved checkpoint.

### Checkpoints

- **Best model**: Automatically saved whenever the validation accuracy improves.
- **Last checkpoint**: Saved at the end of each epoch to allow resuming from the most recent state.

## Model Architecture

The **EfficientViT-B4** model is part of the EfficientViT family, designed for optimal speed and accuracy in vision tasks. This implementation uses custom configuration settings to balance computational efficiency and model performance.

- **Model architecture**: EfficientViT-B4
- **Input size**: 256x256 pixels
- **Pretraining**: The model is trained from scratch, with no initial weights.

## Training Configuration

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 (with exponential decay)
- **Batch Size**: 42
- **Gradient Accumulation**: 3 steps to control memory usage
- **Epochs**: 16
- **Data Augmentation**: Resize, Color Jitter, Random Horizontal Flip, and Normalization

## Resume Pretraining

If pretraining is interrupted, the script will automatically resume from the last checkpoint. The model, optimizer, and scheduler states are restored from the latest saved checkpoint.

## Results and Validation

During pretraining, validation is performed at the end of each epoch to evaluate the model's performance. Metrics such as loss and accuracy are logged and tracked.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
