import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets # For ImageFolder

# Import your ResNet18 model from the specified path
# Please ensure your ./model/resnet18.py exists and contains a resnet18() function
from model.resnet18 import resnet18 

# Define AvgMeter utility class (copied directly for completeness)
class AvgMeter(object):
    """
    Utility class to compute and store the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        """
        Adds a new value to the meter.
        Args:
            val (float): Current value.
            n (int): Number of elements.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


# Configure device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==============================================================================
# Data Loading Function for ImageNet
# ==============================================================================
def load_data(args):
    """
    Loads ImageNet training and validation datasets.
    Args:
        args (argparse.Namespace): Command-line arguments containing batch_size and data_dir.
    Returns:
        tuple: (train_loader, val_loader, num_classes) PyTorch DataLoaders for ImageNet and class count.
    """
    print('==> Preparing ImageNet data..')
    data_base_dir = args.data_dir # Use the data_dir argument

    # ImageNet standard preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224), # ImageNet typically uses 224x224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),       # Scale smaller side to 256
        transforms.CenterCrop(224),   # Crop the center 224x224
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_base_dir, 'train')
    val_dir = os.path.join(data_base_dir, 'val')

    # Check if data directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"ImageNet training directory not found: '{train_dir}'")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet validation directory not found: '{val_dir}'")

    trainset = datasets.ImageFolder(train_dir, transform_train)
    # Using DataLoader for ImageNet training
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    valset = datasets.ImageFolder(val_dir, transform_val)
    # Using DataLoader for ImageNet validation
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    num_classes = len(trainset.classes)
    print(f"Data loaded. Train samples: {len(trainset)}, Val samples: {len(valset)}")
    print(f"Detected {num_classes} classes from ImageNet data.")

    return trainloader, valloader, num_classes

# ==============================================================================
# Training Function
# ==============================================================================
def train_model(model, train_loader, args, val_loader_for_eval, epoch_start_val=0):
    """
    Trains the model for specified epochs.
    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        args (argparse.Namespace): Command-line arguments (lr, epochs, etc.).
        val_loader_for_eval (torch.utils.data.DataLoader): DataLoader for evaluation during training.
        epoch_start_val (int): Starting epoch number (useful for resuming).
    Returns:
        torch.nn.Module: The trained model.
    """
    print(f"\n--- Starting training for {args.epochs - epoch_start_val} epochs (Total: {args.epochs}) ---")
    
    # Set model to training mode
    model.train() 
    
    # Optimizer (Stochastic Gradient Descent with Momentum and Weight Decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4) # Common for ImageNet
    # Learning Rate Scheduler (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Loss Function (Cross Entropy Loss for classification)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0 # Track best accuracy for saving checkpoints
    
    # If resuming, load the best_acc from checkpoint if available
    if args.resume and args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device)
        if 'acc' in checkpoint:
            best_acc = checkpoint['acc']
            print(f"Resumed initial best_acc from checkpoint: {best_acc * 100:.2f}%")
        # If scheduler state needs to be loaded for resume:
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Resumed scheduler state.")
        # Make sure the scheduler's current epoch is aligned for CosineAnnealing
        # This is a common pitfall. If T_max is total epochs, scheduler.step() needs to reflect that.
        for _ in range(epoch_start_val): # Apply scheduler steps for past epochs
            scheduler.step()
        print(f"Adjusted scheduler to epoch {epoch_start_val}. Current LR: {optimizer.param_groups[0]['lr']:.6f}")


    for epoch in range(epoch_start_val, args.epochs):
        print(f'\nTrain Epoch: {epoch+1}/{args.epochs} - Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        loss_meter = AvgMeter()
        acc_meter = AvgMeter()
        
        # Use tqdm for a progress bar during training
        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero gradients before each backward pass
                optimizer.zero_grad()
                # Forward pass
                outputs = model(inputs)
                # Calculate loss
                loss = criterion(outputs, targets.long())
                # Backward pass
                loss.backward()
                # Update model parameters
                optimizer.step()

                # Update metrics
                loss_meter.add(loss.item(), inputs.size(0))
                # ImageNet classification is usually Top-1 accuracy
                # outputs.argmax(dim=-1) gives predicted class index
                acc = (outputs.argmax(dim=-1).long() == targets).float().mean() 
                acc_meter.add(acc.item(), inputs.size(0))

                # Update tqdm progress bar
                pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg * 100})
                pbar.update(1)

        # Step the learning rate scheduler after each epoch
        scheduler.step()
        print(f"Train Epoch {epoch+1} finished. Avg Loss: {loss_meter.avg:.4f}, Avg Acc: {acc_meter.avg * 100:.2f}%")
        
        # Evaluate after each epoch on the validation set to track progress and find best model
        current_loss, current_acc = test_model(model, val_loader_for_eval, f"Epoch {epoch+1} Validation") 
        
        # Save checkpoint if current model has the best accuracy so far
        if current_acc > best_acc:
            print(f"Saving best model with validation accuracy: {current_acc * 100:.2f}% (Previous best: {best_acc * 100:.2f}%)")
            state = {
                'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'acc': current_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), # Save scheduler state
            }
            # Create checkpoint directory if it doesn't exist
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint')
            
            # Save checkpoint with model name, total epochs, and best accuracy in filename
            save_path = f'./checkpoint/ResNet18_ImageNet_epoch{args.epochs}_best_acc_{current_acc*100:.2f}.pth'
            torch.save(state, save_path)
            best_acc = current_acc
        else:
            print(f"Current validation accuracy {current_acc * 100:.2f}% is not better than best {best_acc * 100:.2f}%. Not saving.")


    print("--- Training complete ---")
    return model

# ==============================================================================
# Test Function (for Validation/Final Evaluation)
# ==============================================================================
def test_model(model, data_loader, description="Test"):
    """
    Evaluates model performance on a given data loader (validation or test).
    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        description (str): Description for the progress bar and print statements.
    Returns:
        tuple: (avg_loss, avg_acc) Average loss and accuracy on the dataset.
    """
    # Set model to evaluation mode
    model.eval() 
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Use tqdm for a progress bar during evaluation
        for image_batch, gt_batch in tqdm(data_loader, desc=description):
            image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
            gt_batch = gt_batch.long() # Ensure target type is long

            # Forward pass
            pred_batch = model(image_batch)
            # Calculate loss
            loss = criterion(pred_batch, gt_batch)
            
            # Update metrics
            loss_meter.add(loss.item(), image_batch.size(0))
            acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
            acc_meter.add(acc.item(), image_batch.size(0))
    
    avg_loss = loss_meter.avg
    avg_acc = acc_meter.avg

    print(f"--- {description} Result --- Loss: {avg_loss:.4f}, Accuracy: {avg_acc * 100:.2f}%")
    return avg_loss, avg_acc

# ==============================================================================
# Main Function
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet18 Training')
    parser.add_argument('--data_dir', default='./data/ImageNet2012', type=str, help='Path to ImageNet 2012 dataset root (containing train/val folders)')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for data loaders')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate') # ImageNet often starts with 0.1
    # Increased default epochs for ImageNet training
    parser.add_argument('--epochs', default=90, type=int, help='number of epochs to train') 
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (recommend >= 4 for ImageNet)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # New argument for specifying resume path, made required if --resume is used
    parser.add_argument('--resume_path', type=str, default=None, help='Path to the .pth checkpoint file to resume from. Required if --resume is used.')
    args = parser.parse_args()

    # Input validation for resume: if --resume is used, --resume_path must be provided
    if args.resume and args.resume_path is None:
        parser.error("--resume requires --resume_path to be specified.")
    
    # 1. Load Data
    train_loader, val_loader, num_classes = load_data(args)

    # 2. Instantiate Model
    print("\n--- Initializing ResNet18 model ---")
    # Pass num_classes to your resnet18 model if it's configurable
    net = resnet18(num_classes=num_classes) 
    net = net.to(device)

    start_epoch = 0
    if args.resume:
        # Load checkpoint from specified path
        print(f'==> Resuming from checkpoint: {args.resume_path}..')
        if not os.path.exists(args.resume_path):
            print(f'Error: Checkpoint file not found at {args.resume_path}!')
            exit() # Exit if checkpoint file is not found
        
        # Load checkpoint to CPU first, then transfer to device if needed
        checkpoint = torch.load(args.resume_path, map_location=device)
        
        # Handle DataParallel state_dict if necessary
        # Check if the saved state_dict keys start with 'module.' (from DataParallel)
        # and current model is not DataParallel, or vice-versa.
        # This handles loading checkpoints saved from DataParallel onto a single GPU model,
        # or vice-versa, by stripping/adding 'module.' prefix.
        state_dict = checkpoint['net']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.') and not isinstance(net, nn.DataParallel):
                new_state_dict[k[7:]] = v # Strip 'module.' prefix
            elif not k.startswith('module.') and isinstance(net, nn.DataParallel):
                new_state_dict['module.' + k] = v # Add 'module.' prefix
            else:
                new_state_dict[k] = v
        
        net.load_state_dict(new_state_dict)
            
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        print(f"Resumed from epoch {checkpoint['epoch']} with accuracy {checkpoint['acc'] * 100:.2f}%")

    # If CUDA is available, wrap the model with DataParallel for multi-GPU support
    if device == 'cuda':
        # Check if model is not already DataParallel before wrapping
        if not isinstance(net, nn.DataParallel):
            print("Wrapping model with DataParallel for multi-GPU training.")
            net = torch.nn.DataParallel(net)
        # Enable CuDNN auto-tuner for faster convolution operations
        cudnn.benchmark = True

    # 3. Train the model
    trained_net = train_model(net, train_loader, args, val_loader, epoch_start_val=start_epoch)

    # 4. Evaluate Final Trained Model Performance
    print("\n--- Evaluating Final Trained Model ---")
    final_loss, final_acc = test_model(trained_net, val_loader, "Final Validation")

    print("\n--- ImageNet ResNet18 Training Process Complete ---")
    print(f"Final Validation Accuracy: {final_acc * 100:.2f}%")