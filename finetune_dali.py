import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Removed torchvision.transforms and torchvision.datasets as DALI replaces them
import torchvision.models as models # For loading pre-trained ResNet18

# DALI Imports
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

# Define AvgMeter utility class
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
# DALI ImageNet Pipeline
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
        jpegs, labels = fn.readers.file(
            file_root=os.path.join(self.data_dir, 'train' if self.is_training else 'val'),
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            random_shuffle=self.is_training,
            name="Reader"
        )

        if self.is_training:
            images = fn.decoders.image_random_crop(jpegs, device="mixed", output_type=types.RGB,
                                                   random_aspect_ratio=[0.75, 1.33],
                                                   random_area=[0.08, 1.0])
            images = fn.resize(images, resize_x=self.image_size, resize_y=self.image_size)
            flip_coin = fn.random.coin_flip(probability=0.5)
            images = fn.flip(images, horizontal=flip_coin)
        else:
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
# Data Loader with DALI
# ===========================================
def load_data(args):
    data_base_dir = args.data_dir
    train_dir = os.path.join(data_base_dir, 'train')
    val_dir = os.path.join(data_base_dir, 'val')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"ImageNet training directory not found: '{train_dir}'")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"ImageNet validation directory not found: '{val_dir}'")

    world_size = 1
    local_rank = 0
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()

    train_samples = 1281167
    val_samples = 50000

    train_pipe = ImageNetDaliPipeline(
        data_dir=data_base_dir,
        image_size=224,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        device_id=local_rank,
        shard_id=local_rank,
        num_shards=world_size,
        is_training=True
    )
    train_pipe.build()
    train_loader = DALIClassificationIterator(train_pipe,
                                          reader_name="Reader",
                                          auto_reset=True)

    val_pipe = ImageNetDaliPipeline(
        data_dir=data_base_dir,
        image_size=224,
        batch_size=args.batch_size,
        num_threads=args.num_workers,
        device_id=local_rank,
        shard_id=local_rank,
        num_shards=world_size,
        is_training=False
    )
    val_pipe.build()
    val_loader = DALIClassificationIterator(val_pipe,
                                        reader_name="Reader",
                                        auto_reset=True)

    num_classes = 1000
    print(f"[DALI] Train samples per shard: {train_samples // world_size}, Val samples: {val_samples // world_size}")
    return train_loader, val_loader, num_classes

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
    
    model.train() 
    
    # Optimizer and Scheduler for fine-tuning
    # Typically a smaller learning rate for fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0 
    
    if args.resume and args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device)
        if 'acc' in checkpoint:
            best_acc = checkpoint['acc']
            print(f"Resumed initial best_acc from checkpoint: {best_acc * 100:.2f}%")
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("Resumed scheduler state.")
        for _ in range(epoch_start_val): 
            scheduler.step()
        print(f"Adjusted scheduler to epoch {epoch_start_val}. Current LR: {optimizer.param_groups[0]['lr']:.6f}")


    for epoch in range(epoch_start_val, args.epochs):
        print(f'\nTrain Epoch: {epoch+1}/{args.epochs} - Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        loss_meter = AvgMeter()
        top1_acc_meter = AvgMeter()
        top5_acc_meter = AvgMeter()
        
        train_loader.reset() # DALI迭代器需要手动重置

        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}") as pbar:
            for batch_idx, data in enumerate(train_loader):
                inputs = data[0]['data'].to(device)
                targets = data[0]['label'].squeeze(-1).to(device) # .squeeze(-1) 移除可能的多余维度

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.item(), inputs.size(0))
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                top1_acc_meter.add(acc1.item(), inputs.size(0))
                top5_acc_meter.add(acc5.item(), inputs.size(0))

                pbar.set_postfix({'loss': loss_meter.avg, 
                                  'acc1': top1_acc_meter.avg, 
                                  'acc5': top5_acc_meter.avg})
                pbar.update(1)

        scheduler.step()
        print(f"Train Epoch {epoch+1} finished. Avg Loss: {loss_meter.avg:.4f}, Avg Acc1: {top1_acc_meter.avg:.2f}%, Avg Acc5: {top5_acc_meter.avg:.2f}%")
        
        current_loss, current_top1_acc, current_top5_acc = test_model(model, val_loader, f"Epoch {epoch+1} Validation") 
        
        if current_top1_acc > best_acc:
            print(f"Saving best model with validation Top-1 accuracy: {current_top1_acc:.2f}% (Previous best: {best_acc:.2f}%)")
            state = {
                'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'acc': current_top1_acc, # Store Top-1 for best_acc tracking
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint')
            
            save_path = f'./checkpoint/ResNet18_ImageNet_AvgPool_epoch{args.epochs}_best_acc_{current_top1_acc:.2f}.pth'
            torch.save(state, save_path)
            best_acc = current_top1_acc
        else:
            print(f"Current validation Top-1 accuracy {current_top1_acc:.2f}% is not better than best {best_acc:.2f}%. Not saving.")


    print("--- Training complete ---")
    return model

# ==============================================================================
# Test Function (for Validation/Final Evaluation)
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
    model.eval() 
    loss_meter = AvgMeter()
    top1_acc_meter = AvgMeter()
    top5_acc_meter = AvgMeter() 
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    data_loader.reset() # DALI迭代器需要手动重置

    with torch.no_grad():
        with tqdm(data_loader, desc=description) as pbar:
            for data in pbar:
                image_batch = data[0]['data'].to(device)
                gt_batch = data[0]['label'].squeeze(-1).to(device)
                gt_batch = gt_batch.long()

                pred_batch = model(image_batch)
                loss = criterion(pred_batch, gt_batch)
                
                loss_meter.add(loss.item(), image_batch.size(0))
                
                acc1, acc5 = accuracy(pred_batch, gt_batch, topk=(1, 5))
                top1_acc_meter.add(acc1.item(), image_batch.size(0))
                top5_acc_meter.add(acc5.item(), image_batch.size(0))

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
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet18 Fine-tuning with AvgPool')
    parser.add_argument('--data_dir', default='./data/ImageNet2012', type=str, 
                        help='Path to ImageNet 2012 dataset root (containing train/val folders)')
    parser.add_argument('--batch_size', default=256, type=int, 
                        help='batch size for data loaders')
    parser.add_argument('--lr', default=0.001, type=float, # Smaller LR for fine-tuning
                        help='initial learning rate') 
    parser.add_argument('--epochs', default=20, type=int, # Fine-tuning usually needs fewer epochs
                        help='number of epochs to fine-tune') 
    parser.add_argument('--num_workers', default=24, type=int, 
                        help='number of data loading workers (recommend >= 4 for ImageNet)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', type=str, default=None, 
                        help='Path to the .pth checkpoint file to resume from. Required if --resume is used.')
    args = parser.parse_args()

    if args.resume and args.resume_path is None:
        parser.error("--resume requires --resume_path to be specified.")
    
    # 1. Load Data
    train_loader, val_loader, num_classes = load_data(args)

    # 2. Instantiate and Modify Model
    print("\n--- Loading pre-trained ResNet18 model from torchvision ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # --- Critical modification: Replace MaxPool with AvgPool ---
    print("--- Replacing initial MaxPool layer with AvgPool layer ---")
    model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    # Ensure the final fully connected layer matches the number of classes in your dataset
    # This is important if you are fine-tuning on a different dataset than ImageNet-1K (1000 classes).
    # If fine-tuning on ImageNet-1K, this line can be optionally included for clarity.
    if model.fc.out_features != num_classes:
        print(f"Adjusting final FC layer from {model.fc.out_features} to {num_classes} output features.")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    start_epoch = 0
    if args.resume:
        print(f'==> Resuming from checkpoint: {args.resume_path}..')
        if not os.path.exists(args.resume_path):
            print(f'Error: Checkpoint file not found at {args.resume_path}!')
            exit()
        
        checkpoint = torch.load(args.resume_path, map_location=device)
        state_dict = checkpoint['net']
        
        # Handle DataParallel state_dict if necessary
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.') and not isinstance(model, nn.DataParallel):
                new_state_dict[k[7:]] = v
            elif not k.startswith('module.') and isinstance(model, nn.DataParallel):
                new_state_dict['module.' + k] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} with accuracy {checkpoint['acc'] * 100:.2f}%")

    if device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # 3. Train the model
    trained_net = train_model(model, train_loader, args, val_loader, epoch_start_val=start_epoch)

    # 4. Evaluate Final Trained Model Performance
    print("\n--- Evaluating Final Fine-tuned Model ---")
    final_loss, final_top1_acc, final_top5_acc = test_model(trained_net, val_loader, "Final Fine-tuned Model Validation")

    print("\n--- ResNet18 Fine-tuning Process Complete ---")
    print(f"Final Validation Results (AvgPool in initial layer):")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Top-1 Accuracy: {final_top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {final_top5_acc:.2f}%")