import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# DALI Imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

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

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t() 
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Configure device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ===========================================
# DALI ImageNet Pipeline (for Validation)
# ===========================================
class ImageNetDaliPipeline(Pipeline):
    def __init__(self, data_dir, image_size, batch_size, num_threads, device_id, shard_id, num_shards, is_training):
        super().__init__(batch_size, num_threads, device_id, seed=42)
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_training = is_training
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def define_graph(self):
        # The reader should point to the validation set
        jpegs, labels = fn.readers.file(
            file_root=os.path.join(self.data_dir, 'val'), # Always 'val' for testing
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            random_shuffle=False, # No shuffle for validation
            name="Reader"
        )

        # Validation preprocessing
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_shorter=256)
        images = fn.crop(images, crop_h=self.image_size, crop_w=self.image_size)

        images = fn.crop_mirror_normalize(images,
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          mean=self.mean,
                                          std=self.std)
        return images, labels

# ===========================================
# Data Loader with DALI (for Validation only)
# ===========================================
def load_val_data_dali(args):
    data_base_dir = args.data_dir
    val_dir = os.path.join(data_base_dir, 'val')

    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet validation directory not found: '{val_dir}'")

    world_size = 1
    local_rank = 0
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()

    val_samples = 50000 # Standard ImageNet val samples

    val_pipe = ImageNetDaliPipeline(
        data_dir=data_base_dir,
        image_size=224, # Standard input size for ResNet
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        device_id=local_rank,
        shard_id=local_rank,
        num_shards=world_size,
        is_training=False # Always False for validation
    )
    val_pipe.build()
    val_loader = DALIClassificationIterator(val_pipe,
                                         reader_name="Reader",
                                         auto_reset=True,
                                         last_batch_policy=LastBatchPolicy.PARTIAL) # Allow partial batches at the end

    num_classes = 1000 # ImageNet has 1000 classes
    print(f"[DALI] Val samples per shard: {val_samples // world_size}")
    return val_loader, num_classes

# ==============================================================================
# Test Function (for Validation/Final Evaluation) - Adjusted for DALI
# ==============================================================================
def test_model(model, data_loader, description="Test"):
    """
    Evaluates model performance on a given DALI data loader,
    calculating both Top-1 and Top-5 accuracy.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_loader (DALIClassificationIterator): DALI DataLoader for evaluation data.
        description (str): Description for the progress bar and print statements.
    Returns:
        tuple: (avg_loss, top1_acc, top5_acc) Average loss, Top-1 accuracy, and Top-5 accuracy.
    """
    model.eval() 
    loss_meter = AvgMeter()
    top1_acc_meter = AvgMeter()
    top5_acc_meter = AvgMeter() 
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    data_loader.reset() # DALI iterator needs to be manually reset for each epoch/run

    with torch.no_grad():
        # DALI iterator doesn't have __len__ directly for tqdm progress,
        # but we can get it from the pipeline's epoch size or total number of batches
        total_batches = data_loader._size // data_loader.batch_size # Approximate total batches

        with tqdm(total=total_batches, desc=description) as pbar:
            for i, data in enumerate(data_loader):
                # DALI returns a list of dictionaries, typically with one dictionary for one output.
                # 'data' key holds the images, 'label' key holds the labels.
                image_batch = data[0]['data'].to(device)
                gt_batch = data[0]['label'].squeeze(-1).to(device) # .squeeze(-1) if label has extra dim
                gt_batch = gt_batch.long() # Ensure target type is long

                pred_batch = model(image_batch)
                loss = criterion(pred_batch, gt_batch)
                
                loss_meter.add(loss.item(), image_batch.size(0))
                
                acc1, acc5 = accuracy(pred_batch, gt_batch, topk=(1, 5))
                top1_acc_meter.add(acc1.item(), image_batch.size(0))
                top5_acc_meter.add(acc5.item(), image_batch.size(0))

                pbar.set_postfix({'loss': loss_meter.avg, 
                                  'acc1': top1_acc_meter.avg, 
                                  'acc5': top5_acc_meter.avg})
                pbar.update(1)
    
    avg_loss = loss_meter.avg
    top1_acc = top1_acc_meter.avg
    top5_acc = top5_acc_meter.avg

    print(f"--- {description} Result --- Loss: {avg_loss:.4f}, Top-1 Accuracy: {top1_acc:.2f}%, Top-5 Accuracy: {top5_acc:.2f}%")
    return avg_loss, top1_acc, top5_acc

# ==============================================================================
# Main Function for Testing
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet18 Evaluation with DALI')
    parser.add_argument('--data_dir', default='./data/ImageNet2012', type=str, 
                        help='Path to ImageNet 2012 dataset root (containing val folder)')
    parser.add_argument('--batch_size', default=256, type=int, 
                        help='batch size for data loaders')
    parser.add_argument('--num_workers', default=24, type=int, 
                        help='number of DALI pipeline threads (recommend >= 4 for ImageNet)')
    parser.add_argument('--load_checkpoint', type=str, default=None, 
                        help='Path to a custom model checkpoint (.pth) to load for testing. '
                             'If not specified, the default ImageNet pre-trained ResNet18 will be used.')
    # Add an argument to include AvgPool replacement
    parser.add_argument('--avgpool_replace', action='store_true', 
                        help='Replace the initial MaxPool layer with AvgPool for ResNet18. '
                             'Use this if your checkpoint was trained with this modification.')
    args = parser.parse_args()

    # Enable cuDNN auto-tuner
    cudnn.benchmark = True

    # 1. Load Data (Validation Set Only) using DALI
    val_loader, num_classes = load_val_data_dali(args)

    # 2. Instantiate and Load Model
    print("\n--- Initializing ResNet18 model ---")
    model = models.resnet18(weights=None) # Start with no weights to load custom or pre-trained later

    # Apply AvgPool replacement if specified
    if args.avgpool_replace:
        print("--- Replacing initial MaxPool layer with AvgPool layer ---")
        model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    # Adjust final FC layer if the number of classes is different (e.g., if fine-tuned on a smaller dataset)
    # For ImageNet, num_classes is 1000, so this might not change.
    if model.fc.out_features != num_classes:
        print(f"Adjusting final FC layer from {model.fc.out_features} to {num_classes} output features.")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.load_checkpoint:
        print(f"\n--- Loading model from custom checkpoint: {args.load_checkpoint} ---")
        if not os.path.exists(args.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: '{args.load_checkpoint}'")
        
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        
        # Robustly extract state_dict from checkpoint
        state_dict = None
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: # Common PyTorch Lightning/custom trainer
                state_dict = checkpoint['state_dict']
            elif 'net' in checkpoint: # As observed in your training code
                state_dict = checkpoint['net']
            else: # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else: # Assume checkpoint is directly the state_dict
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
        
        # Load the state dictionary
        model.load_state_dict(new_state_dict)
        print("Custom checkpoint loaded successfully.")
    else:
        print("\n--- Loading default pre-trained ResNet18 model from torchvision (ImageNet-1K V1 weights) ---")
        # Load weights explicitly, ensuring no AvgPool replacement unless specified
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Re-apply AvgPool replacement if needed after loading default weights
        if args.avgpool_replace:
             print("--- Re-applying initial MaxPool layer with AvgPool layer after loading pre-trained weights ---")
             model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        print("Default pre-trained model loaded.")
    
    model = model.to(device)

    # Use DataParallel if multiple GPUs are available
    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    # 3. Evaluate the model on the Validation Set
    print("\n--- Starting Evaluation ---")
    final_loss, final_top1_acc, final_top5_acc = test_model(model, val_loader, "Model Evaluation")

    print("\n--- Evaluation Complete ---")
    print(f"Final Model Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Top-1 Accuracy: {final_top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {final_top5_acc:.2f}%")