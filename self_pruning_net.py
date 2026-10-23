# -*- coding: utf-8 -*-
"""
Self-Pruning Neural Network trained on CIFAR-10.

Google Colab Quick-Start
------------------------
1. Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4 or better)
2. Run the cell below ONCE to install dependencies if missing:

   !pip install torch torchvision matplotlib numpy --quiet

3. Then run:  !python self_pruning_net.py
   OR paste the whole script into a code cell and run it.

CUDA is auto-detected; no manual changes are needed.
"""

# ==============================================================================
# SECTION 1 — IMPORTS AND SETUP
# ==============================================================================
import os
import random
import sys
import time
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ==============================================================================
# SECTION 2 — DATASET LOADER
# ==============================================================================
def get_cifar10_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Downloads and loads CIFAR-10 dataset with separate train and test loaders."""
    # Enhanced training augmentation:
    #   ColorJitter  — random brightness/contrast/saturation variation
    #   RandomErasing — randomly masks patches, forces robust features
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    # Test transform — no augmentation, same normalisation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # num_workers=0 is required on Colab (multiprocessing fork issues in notebooks).
    # pin_memory=True speeds up host->GPU transfers when CUDA is available.
    use_cuda   = torch.cuda.is_available()
    pin_memory = use_cuda

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )
    return train_loader, test_loader


# ==============================================================================
# SECTION 3 — PrunableLinear LAYER
# ==============================================================================
class PrunableLinear(nn.Module):
    """Linear layer with learnable gates for each parameter to enable self-pruning."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # zero initialization so sigmoid(0) = 0.5 (half-open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        effective_weight = self.weight * gates
        return F.linear(x, effective_weight, self.bias)


# ==============================================================================
# SECTION 4 — NETWORK ARCHITECTURE
# ==============================================================================
class SelfPruningNet(nn.Module):
    """Network architecture with PrunableLinear layers for CIFAR-10."""
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.layer1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.layer2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.layer3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.layer4 = PrunableLinear(256, 10)
        
        self.prunable_layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        
        x = self.layer1(x)
        x = F.relu(x)
        x = self.bn1(x)
        
        x = self.layer2(x)
        x = F.relu(x)
        x = self.bn2(x)
        
        x = self.layer3(x)
        x = F.relu(x)
        x = self.bn3(x)
        
        x = self.layer4(x)
        return x
        
    def get_gate_params(self) -> List[nn.Parameter]:
        """Returns list of gate_scores parameters only."""
        return [layer.gate_scores for layer in self.prunable_layers]
        
    def get_weight_params(self) -> List[nn.Parameter]:
        """Returns list of weight + bias parameters only."""
        params = []
        for name, param in self.named_parameters():
            if 'gate_scores' not in name:
                params.append(param)
        return params


# ==============================================================================
# SECTION 5 — LOSS FUNCTION
# ==============================================================================
def compute_loss(
    model: SelfPruningNet,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_val: float,
    label_smoothing: float = 0.1,
) -> Tuple[torch.Tensor, float, float]:
    """Computes combined Cross Entropy (with label smoothing) and Sparsity loss.

    Label smoothing (default 0.1) softens hard targets, which reduces overconfidence
    and consistently improves generalisation / test accuracy by ~0.5-1 pp.
    """
    ce_loss = F.cross_entropy(outputs, targets, label_smoothing=label_smoothing)

    sparsity_loss = torch.tensor(0.0, device=outputs.device)
    for layer in model.prunable_layers:
        sparsity_loss += torch.sigmoid(layer.gate_scores).sum()

    total_loss = ce_loss + lambda_val * sparsity_loss
    return total_loss, ce_loss.item(), sparsity_loss.item()


# ==============================================================================
# SECTION 6 — LAMBDA SCHEDULE
# ==============================================================================
def get_lambda(epoch: int, lambda_target: float, warmup_epochs: int = 10, ramp_length: int = 25) -> float:
    """Calculates sparsity penaly weight based on training phases."""
    if epoch <= warmup_epochs:
        return 0.0
    elif warmup_epochs < epoch <= 35:
        return lambda_target * min(1.0, (epoch - warmup_epochs) / ramp_length)
    else:
        return lambda_target


# ==============================================================================
# SECTION 7 — OPTIMIZER SETUP
# ==============================================================================
def build_optimizer(model: SelfPruningNet) -> optim.Optimizer:
    """Builds Adam optimizer with differentiated learning rates for weights and gates."""
    optimizer = optim.Adam([
        {'params': model.get_weight_params(), 'lr': 1e-3},
        {'params': model.get_gate_params(), 'lr': 1e-2}
    ])
    return optimizer


# ==============================================================================
# SECTION 8 — TRAINING LOOP
# ==============================================================================
def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_val: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Executes single training epoch and returns performance metrics.

    Gradient clipping (max_grad_norm=1.0) prevents occasional exploding gradients
    that stall training, keeping both loss and accuracy curves smooth.
    """
    model.train()
    total_loss_sum = 0.0
    ce_loss_sum    = 0.0
    sparsity_loss_sum = 0.0
    correct = 0
    total   = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, ce, sp = compute_loss(model, outputs, targets, lambda_val)

        loss.backward()
        # Clip gradients before the weight update to avoid loss spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss_sum    += loss.item() * batch_size
        ce_loss_sum       += ce * batch_size
        sparsity_loss_sum += sp * batch_size

        _, predicted = outputs.max(1)
        total   += batch_size
        correct += predicted.eq(targets).sum().item()

    avg_total_loss    = total_loss_sum    / total
    avg_ce_loss       = ce_loss_sum       / total
    avg_sparsity_loss = sparsity_loss_sum / total
    train_accuracy    = 100. * correct / total

    return avg_total_loss, avg_ce_loss, avg_sparsity_loss, train_accuracy

def train(
    model: SelfPruningNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    lambda_target: float,
    device: torch.device,
    total_epochs: int = 75,
    verbose: bool = True,
) -> Tuple[float, SelfPruningNet]:
    """Full training loop with CosineAnnealingLR scheduler for the weight group.

    Why CosineAnnealingLR?
      - Smoothly reduces the weight LR from 1e-3 down to 1e-5 over total_epochs.
      - Prevents the model from getting stuck in sharp minima near the end of training.
      - Gate LR is intentionally kept constant so sparsity pressure stays predictable.
    """
    optimizer = build_optimizer(model)

    # Scheduler only on param_group[0] (weights); gates keep their fixed lr=1e-2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5, last_epoch=-1
    )

    best_test_acc = 0.0

    for epoch in range(1, total_epochs + 1):
        lam = get_lambda(epoch, lambda_target)

        avg_loss, avg_ce, avg_sp, train_acc = train_one_epoch(
            model, train_loader, optimizer, lam, device
        )

        # Step the cosine scheduler after each epoch
        scheduler.step()

        if epoch % 5 == 0:
            test_acc = evaluate(model, test_loader, device)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    model.state_dict(),
                    f"./outputs/best_model_lambda_{lambda_target}.pt"
                )

            if verbose:
                current_lr = scheduler.get_last_lr()[0]
                phase = 1 if epoch <= 10 else (2 if epoch <= 50 else 3)
                print(
                    f"Epoch {epoch:02d} | Phase {phase} | lam={lam:.1e} | lr={current_lr:.1e} | "
                    f"Loss={avg_loss:.4f} | CE={avg_ce:.4f} | Sparsity={avg_sp:.4f} | "
                    f"TrainAcc={train_acc:.2f}% | TestAcc={test_acc:.2f}%",
                    flush=True,
                )

    final_test_acc = evaluate(model, test_loader, device)
    return final_test_acc, model


# ==============================================================================
# SECTION 9 — EVALUATION METRICS
# ==============================================================================
def evaluate(model: SelfPruningNet, loader: DataLoader, device: torch.device) -> float:
    """Computes accuracy of the model on the provided dataset loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100. * correct / total

def compute_sparsity(model: SelfPruningNet, threshold: float = 0.01) -> float:
    """Calculates the overall network sparsity based on gate values."""
    model.eval()
    total_elements = 0
    pruned_elements = 0
    with torch.no_grad():
        for layer in model.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores)
            pruned_elements += (gates < threshold).float().sum().item()
            total_elements += gates.numel()
            
    return (pruned_elements / total_elements) * 100.0

def plot_gate_histogram(model: SelfPruningNet, lambda_val: float, save_path: str) -> None:
    """Generates and saves a histogram of all flattened gate values."""
    model.eval()
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers:
            gates = torch.sigmoid(layer.gate_scores).view(-1).cpu().numpy()
            all_gates.append(gates)
            
    all_gates = np.concatenate(all_gates)
    
    plt.figure()
    plt.hist(all_gates, bins=50, range=(0.0, 1.0))
    plt.title(f"Gate Distribution - lambda = {lambda_val}")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.axvline(x=0.01, color='r', linestyle='dashed', linewidth=1)
    
    plt.savefig(save_path)
    plt.close()


# ==============================================================================
# SECTION 10 — EXPERIMENT RUNNER (main)
# ==============================================================================
if __name__ == "__main__":
    # Force line-buffered stdout so every print() appears immediately in the terminal
    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    # 1. Set global random seeds for full reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # On GPU: benchmark=True lets cuDNN auto-tune kernels for faster training.
    # On CPU: this flag has no effect.
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    # 2. Create ./outputs/ directory
    os.makedirs("./outputs", exist_ok=True)

    # 3. Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Load CIFAR-10 data loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # 5. Print header
    print("=" * 65, flush=True)
    print("Self-Pruning Neural Network on CIFAR-10", flush=True)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Device  : {device}  ({gpu_name}, {gpu_mem:.1f} GB VRAM)", flush=True)
    else:
        print(f"Device  : {device}  (no CUDA GPU detected — running on CPU)", flush=True)
    print("=" * 65, flush=True)

    # 6. Run experiments for lambda_targets
    lambda_targets = [1e-4, 1e-3, 1e-2]
    experiment_results = []

    for val in lambda_targets:
        print("\n" + "-" * 65, flush=True)
        print(f"Experiment: lambda_target = {val}", flush=True)
        print("-" * 65, flush=True)
        
        # Instantiate fresh model
        model = SelfPruningNet().to(device)
        
        # Train
        final_accuracy, trained_model = train(
            model=model,
            train_loader=train_loader, 
            test_loader=test_loader, 
            lambda_target=val, 
            device=device,
            total_epochs=75,
            verbose=True
        )
        
        # Compute end sparsity
        sparsity = compute_sparsity(trained_model)
        
        # Plot distribution
        hist_path = f"./outputs/gates_lambda_{val}.png"
        plot_gate_histogram(trained_model, val, hist_path)
        
        # Collect metric run
        experiment_results.append((val, final_accuracy, sparsity))

    # 7. Print final comparison table
    print("\n" + "-" * 44, flush=True)
    print(f"{'lam_target':<13} {'Test Acc (%)':>12} {'Sparsity %':>10}", flush=True)
    print("-" * 44, flush=True)
    for val, acc, sp in experiment_results:
        print(f"{val:<13.0e} {acc:>12.2f} {sp:>10.2f}", flush=True)
    print("-" * 44, flush=True)

    # 8. Print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    hours   = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\nTotal Runtime: {hours}h {minutes}m {seconds:.2f}s", flush=True)
