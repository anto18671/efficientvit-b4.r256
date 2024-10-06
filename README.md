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

To start the pretraining process, make sure you have the following prerequisites:

### Prerequisites
1. **GPU Support**: The pretraining is optimized to run on systems with NVIDIA GPUs. Ensure CUDA and the necessary drivers are installed on your machine.
   - CUDA Version: 12.4 (or compatible version)
   - CuDNN: Version 9

2. **Environment Setup**:
   - Ensure the correct version of **PyTorch** with GPU support is installed.
   - Your system should have enough GPU memory to handle the specified batch size. Modify the batch size if necessary.

3. **Hugging Face Authentication**:
   - You will need to authenticate with Hugging Face to access the ImageNet-1k dataset. Set your Hugging Face token in the environment:

   ```bash
   export HUGGINGFACE_TOKEN=<your_huggingface_token>
   ```

### Starting Pretraining

Once the environment is set up, and the GPU is ready, run the `pre.py` script to begin pretraining:

```bash
python pre.py
```

This script will:
- Initialize the **EfficientViT-B4** model.
- Set up the data pipelines with transformations (resizing, augmentation, normalization).
- Configure the optimizer (AdamW) and the learning rate scheduler.
- Start pretraining from scratch or resume from the last saved checkpoint if any.

### Running in a Docker Environment

If you're using Docker for pretraining, follow these steps:

1. **Pull the Docker Image**:

   ```bash
   docker pull ghcr.io/anto18671/efficientvit-b4.r256:latest
   ```

2. **Run the Docker Container with GPU Support**:

   ```bash
   docker run --gpus all --env HUGGINGFACE_TOKEN=<your_huggingface_token> ghcr.io/anto18671/efficientvit-b4.r256:latest
   ```

Ensure that the Docker setup has GPU support enabled. Use the `--gpus all` flag to allow Docker to utilize the available GPUs.

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
- **Batch Size**: 42 (adjustable based on GPU memory)
- **Gradient Accumulation**: 3 steps to control memory usage
- **Epochs**: 16
- **Data Augmentation**: Resize, Color Jitter, Random Horizontal Flip, and Normalization

## Resume Pretraining

If pretraining is interrupted, the script will automatically resume from the last checkpoint. The model, optimizer, and scheduler states are restored from the latest saved checkpoint.

## Results and Validation

During pretraining, validation is performed at the end of each epoch to evaluate the model's performance. Metrics such as loss and accuracy are logged and tracked.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
