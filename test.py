import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models # For loading pre-trained ResNet18

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

# Function to calculate Top-K accuracy
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get topk predictions
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() # Transpose to match target dimensions

        # Compare predictions with target
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Configure device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==============================================================================
# Data Loading Function for ImageNet Validation Set
# ==============================================================================
def load_val_data(args):
    """
    Loads ImageNet validation dataset.
    Args:
        args (argparse.Namespace): Command-line arguments containing batch_size and data_dir.
    Returns:
        tuple: (val_loader, num_classes) PyTorch DataLoader for ImageNet validation set and class count.
    """
    print('==> Preparing ImageNet validation data..')
    data_base_dir = args.data_dir # Use the data_dir argument

    # ImageNet standard preprocessing for validation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_val = transforms.Compose([
        transforms.Resize(256),       # Scale smaller side to 256
        transforms.CenterCrop(224),   # Crop the center 224x224
        transforms.ToTensor(),
        normalize,
    ])

    val_dir = os.path.join(data_base_dir, 'val')

    # Check if data directory exists
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet validation directory not found: '{val_dir}'")

    valset = datasets.ImageFolder(val_dir, transform_val)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    num_classes = len(valset.classes)
    print(f"Data loaded. Validation samples: {len(valset)}")
    print(f"Detected {num_classes} classes from ImageNet data.")

    return valloader, num_classes

# ==============================================================================
# Test Function (for Validation/Final Evaluation) - Modified for Top-1 & Top-5
# ==============================================================================
def test_model(model, data_loader, description="Test"):
    """
    Evaluates model performance on a given data loader (validation or test),
    calculating both Top-1 and Top-5 accuracy.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        description (str): Description for the progress bar and print statements.
    Returns:
        tuple: (avg_loss, top1_acc, top5_acc) Average loss, Top-1 accuracy, and Top-5 accuracy.
    """
    # Set model to evaluation mode
    model.eval() 
    loss_meter = AvgMeter()
    top1_acc_meter = AvgMeter()
    top5_acc_meter = AvgMeter() # New meter for Top-5 accuracy
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    # Disable gradient calculation for inference
    with torch.no_grad():
        # Use tqdm for a progress bar during evaluation
        with tqdm(data_loader, desc=description) as pbar: # Define pbar here
            for image_batch, gt_batch in pbar:
                image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
                gt_batch = gt_batch.long() # Ensure target type is long

                # Forward pass
                pred_batch = model(image_batch)
                # Calculate loss
                loss = criterion(pred_batch, gt_batch)
                
                # Update metrics
                loss_meter.add(loss.item(), image_batch.size(0))
                
                # Calculate Top-1 and Top-5 accuracy
                acc1, acc5 = accuracy(pred_batch, gt_batch, topk=(1, 5))
                top1_acc_meter.add(acc1.item(), image_batch.size(0))
                top5_acc_meter.add(acc5.item(), image_batch.size(0))

                # Update tqdm progress bar - NOW INSIDE THE 'with' BLOCK
                pbar.set_postfix({'loss': loss_meter.avg, 
                                  'acc1': top1_acc_meter.avg, 
                                  'acc5': top5_acc_meter.avg})
    
    avg_loss = loss_meter.avg
    top1_acc = top1_acc_meter.avg
    top5_acc = top5_acc_meter.avg

    print(f"--- {description} Result --- Loss: {avg_loss:.4f}, Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")
    return avg_loss, top1_acc, top5_acc

# ==============================================================================
# Main Function
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet18 Model Evaluation and Saving')
    parser.add_argument('--data_dir', default='./data/ImageNet2012', type=str, help='Path to ImageNet 2012 dataset root (containing train/val folders)')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size for data loaders')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers (recommend >= 4 for ImageNet)')
    parser.add_argument('--save_model', action='store_true', help='Save the loaded model after evaluation')
    parser.add_argument('--save_path', type=str, default='./checkpoint/resnet18_imagenet_pretrained.pth', help='Path to save the pre-trained model')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a custom model checkpoint (.pth) to load for testing. If not specified, the default ImageNet pre-trained ResNet18 will be used.')
    parser.add_argument('--avgpool_replace', action='store_true', help='Replace the initial MaxPool layer with AvgPool for ResNet18. Use this if your checkpoint was trained with this modification.')
    args = parser.parse_args()

    # 1. Load Data (Validation Set Only)
    val_loader, num_classes = load_val_data(args)

    # 2. Load Model (Pre-trained or Custom Checkpoint)
    if args.load_checkpoint:
        print(f"\n--- Loading model from custom checkpoint: {args.load_checkpoint} ---")
        # Load a base ResNet18 model first to ensure correct architecture
        model = models.resnet18(weights=None) # Initialize with no pre-trained weights

        # Apply AvgPool replacement if specified
        if args.avgpool_replace:
            print("--- Replacing initial MaxPool layer with AvgPool layer ---")
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        # Load state_dict from checkpoint
        if not os.path.exists(args.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: '{args.load_checkpoint}'")
        
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        
        # --- MODIFIED LOGIC HERE ---
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'net' in checkpoint: # This is the key observed in your traceback
                state_dict = checkpoint['net']
            else:
                # If it's a dict but neither 'state_dict' nor 'net' is found,
                # assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            # If checkpoint is not a dict, assume it's directly the state_dict
            state_dict = checkpoint

        if state_dict is None:
            raise ValueError(f"Could not find model state_dict in the checkpoint '{args.load_checkpoint}'. "
                             "Expected a dictionary with 'state_dict' or 'net' key, or the state_dict directly.")

        # Remove 'module.' prefix if saved from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # remove 'module.' prefix
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print("Custom checkpoint loaded successfully.")
    else:
        print("\n--- Loading default pre-trained ResNet18 model from torchvision ---")
        # Using the recommended way to load ImageNet-1K V1 weights
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        print("Default pre-trained model loaded.")
    
    # Ensure the model is moved to the correct device
    model = model.to(device)

    # If you have multiple GPUs, you can wrap it with DataParallel (though not strictly necessary for inference)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # 3. Evaluate the model on the Validation Set
    print("\n--- Evaluating Model on ImageNet Validation Set ---")
    val_loss, top1_acc, top5_acc = test_model(model, val_loader, "ImageNet ResNet18 Validation")

    print("\n--- Evaluation Complete ---")
    print(f"ResNet18 Final Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    # Note: torchvision's reported accuracy for ResNet18 IMAGENET1K_V1 is 69.758% Top-1 and 89.078% Top-5.
    # Your result might vary slightly due to data loading, specific PyTorch/CUDA versions, etc.

    # 4. Save the Model (optional)
    if args.save_model:
        print(f"\n--- Saving the current model to {args.save_path} ---")
        # Create checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
        # Save the model's state_dict
        # If the model was wrapped in DataParallel, unwrap it before saving
        save_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(save_state_dict, args.save_path)
        print("Model saved successfully.")
    else:
        print("\n--- Skipping model saving (use --save_model to enable) ---")