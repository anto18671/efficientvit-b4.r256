# Import necessary libraries
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from datasets import load_dataset
import torch.nn.functional as F
from torch.optim import AdamW
from timm import create_model
from collections import deque
from tqdm import tqdm
import torchsummary
import torch
import os

# Dataset class to handle loading and transforming images
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.dataset)

    # Get a specific sample by index, apply transformation if provided
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Collate function to stack and combine batches
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

# Training loop for one epoch
def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    # Using deques to store the last 128 loss and accuracy values for moving averages
    loss_window = deque(maxlen=accumulation_steps * 128)
    acc_window = deque(maxlen=accumulation_steps * 128)
    
    progress_bar = tqdm(total=len(dataloader)//accumulation_steps, desc="Training", leave=True)
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        
        # Accumulate gradients and scale loss
        loss = loss / accumulation_steps
        loss.backward()

        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Calculate batch-wise loss and accuracy
        batch_loss = loss.item() * accumulation_steps
        batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
        loss_window.append(batch_loss)
        acc_window.append(batch_acc)

        # Perform optimizer step every `accumulation_steps` batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update progress bar with moving averages
            avg_loss = sum(loss_window) / len(loss_window)
            avg_acc = sum(acc_window) / len(acc_window)
            progress_bar.update(1)
            progress_bar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'avg_acc': f'{avg_acc:.3f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.12f}',
            })

    # Handle last batch if it's not divisible by accumulation_steps
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        scheduler.step()

    progress_bar.close()
    return total_loss / len(dataloader), 100. * correct / total

# Validation loop to check model performance on validation data
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar with current loss and accuracy
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1), 
                'acc': 100. * correct / total
            })

    return total_loss / len(dataloader), 100. * correct / total

# Function to save model checkpoints
def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_val_acc, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'best_val_acc': best_val_acc
    }, filename)

# Function to load model checkpoints if they exist
def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        val_acc = checkpoint['val_acc']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return start_epoch, val_acc, best_val_acc
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 0, 0

# Main function for setting up data, model, and training loop
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets (ImageNet-1K)
    train_dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="train")
    val_dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")

    # Shuffle the datasets
    train_dataset = train_dataset.shuffle()
    val_dataset = val_dataset.shuffle()

    # Define transformations for data augmentation and normalization
    transform = v2.Compose([
        v2.Resize((256, 256)),
        v2.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.075),
        v2.RandomHorizontalFlip(),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Wrap the datasets with custom dataset class and transformations
    train_dataset = ImageDataset(train_dataset, transform=transform)
    val_dataset = ImageDataset(val_dataset, transform=transform)

    # DataLoader for batching and parallel data loading
    batch_size = 42
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn, pin_memory=True)

    # Create the EfficientViT model with custom configurations
    model = create_model('efficientvit_b3.r256_in1k', widths=(48, 96, 192, 384, 768), 
                         head_dim=48, head_widths=(3456, 3840), pretrained=False, 
                         num_classes=1000, drop_rate=0.0).to(device)
    
    # Print model summary for clarity
    torchsummary.summary(model, (3, 256, 256))

    # AdamW optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Training parameters
    num_epochs = 16
    accumulation_steps = 3
    total_steps = len(train_loader)// accumulation_steps
    print(f"Total steps per epoch: {total_steps}")

    # Calculate exponential learning rate decay
    decay_rate = 0.8 ** (1 / total_steps)
    print(f"Decay rate: {decay_rate}")

    # Initialize the ExponentialLR scheduler
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)

    # Directory to save checkpoints
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_filename = os.path.join(checkpoint_dir, 'best_model.pth')
    last_checkpoint_filename = os.path.join(checkpoint_dir, 'last_checkpoint.pth')

    # Load checkpoint if available, otherwise start from scratch
    start_epoch, val_acc, best_val_acc = load_checkpoint(model, optimizer, scheduler, last_checkpoint_filename)

    # Main training loop for multiple epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, accumulation_steps)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate the model
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the latest checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_val_acc, last_checkpoint_filename)
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_val_acc, best_model_filename)
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
    print("Training completed!")

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
