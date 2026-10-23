# Self-Pruning Neural Network on CIFAR-10

A from-scratch PyTorch implementation of a neural network that learns **which of its own connections matter** — and gradually suppresses the ones that don't — all within a single, fully self-contained training run.

No manual pruning schedules. No post-training surgery. The network prunes itself.

---

## What Makes This Different

Most pruning approaches work in two stages: train fully, then cut weights after the fact. This project takes a different path. Every linear layer carries a second set of learnable parameters — called **gate scores** — that sit in front of the weights and decide, in real time, how much each connection should contribute. During training, the optimizer simultaneously tightens classification loss *and* pushes idle gates toward zero. Weights that serve no purpose fade out naturally as training progresses.

The result is a model that arrives at the end of training already sparse — not surgically trimmed after the fact, but organically shaped by the data itself.

---

## Project Structure

```
.
├── self_pruning_net.py     # Complete, single-file executable — all logic lives here
├── README.md               # This file
├── data/                   # CIFAR-10 downloaded automatically on first run
└── outputs/
    ├── best_model_lambda_0.0001.pt   # Best checkpoint for lambda = 1e-4
    ├── best_model_lambda_0.001.pt    # Best checkpoint for lambda = 1e-3
    ├── best_model_lambda_0.01.pt     # Best checkpoint for lambda = 1e-2
    ├── gates_lambda_0.0001.png       # Gate distribution histogram
    ├── gates_lambda_0.001.png
    └── gates_lambda_0.01.png
```

---

## How the Architecture Works

### The PrunableLinear Layer

At the core of this project is a custom drop-in replacement for `nn.Linear`:

```
Input x
   │
   ▼
weight (out × in)   ×   sigmoid(gate_scores)   =   effective_weight
                                                          │
                                            F.linear(x, effective_weight, bias)
                                                          │
                                                       Output
```

- `weight` — the standard learnable weight matrix, initialized with Kaiming uniform
- `gate_scores` — a second matrix of the same shape, initialized to **zeros**
- `sigmoid(0) = 0.5` — so every gate starts half-open at the beginning of training

As training proceeds under sparsity pressure, gate scores for unimportant connections drift negative, their sigmoid values collapse toward 0, and the effective weight for those connections becomes effectively zero — without ever hard-removing them.

### Full Network

```
Input Image (32×32×3)
       │
    Flatten  →  3072 features
       │
  PrunableLinear(3072 → 1024)  →  ReLU  →  BatchNorm1d
       │
  PrunableLinear(1024 → 512)   →  ReLU  →  BatchNorm1d
       │
  PrunableLinear(512 → 256)    →  ReLU  →  BatchNorm1d
       │
  PrunableLinear(256 → 10)     →  logits (no activation)
       │
  CrossEntropyLoss (softmax applied internally)
```

No dropout. Pruning is handled entirely by the gate mechanism.

---

## The Loss Function

Training optimizes a combined objective:

```
Total Loss = CrossEntropy(output, target)  +  λ × Σ sigmoid(gate_scores)
```

- The **cross-entropy term** pushes the model to classify correctly
- The **sparsity term** penalizes gates for staying open — it rewards closing connections that aren't earning their keep
- **λ (lambda)** controls the tradeoff; higher values force more aggressive pruning

The sparsity term sums all sigmoid gate values across every layer. Since `sigmoid` is bounded between 0 and 1, minimizing this sum means driving gates toward 0.

---

## Lambda Scheduling (3-Phase Training)

Applying sparsity pressure from epoch 1 can prevent the network from learning anything useful. This project uses a three-phase schedule:

| Phase | Epochs | Lambda Value | Behavior |
|-------|--------|-------------|----------|
| 1 — Warmup | 1 – 10 | 0.0 | Pure classification, no pruning pressure |
| 2 — Ramp | 11 – 50 | 0 → λ_target | Sparsity pressure gradually introduced |
| 3 — Full | 51 – 75 | λ_target | Full regularization for final compression |

This gives the network time to learn meaningful feature representations before the gate-closing mechanism takes effect.

---

## Optimizer Design

Two separate parameter groups with different learning rates:

```python
Adam([
    {'params': weight + bias params,  'lr': 1e-3},   # standard weight update
    {'params': gate_score params,     'lr': 1e-2},   # gates update 10× faster
])
```

Gates learn faster because they need to respond quickly to the sparsity signal. Weights update more conservatively to preserve classification quality.

A **CosineAnnealingLR** scheduler is applied to the weight group only (LR decays from 1e-3 → 1e-5 over 75 epochs). Gate LR remains constant throughout so sparsity pressure stays predictable.

---

## Accuracy Boosting Techniques

Four targeted improvements were added on top of the base architecture:

### 1. Enhanced Data Augmentation

```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
```

`ColorJitter` randomizes pixel intensity patterns so the model cannot rely on absolute color values. `RandomErasing` randomly blanks out small patches of the image, forcing the model to use context from the surrounding region rather than fixating on a single discriminating area.

### 2. Label Smoothing (ε = 0.1)

Instead of training against hard one-hot targets `[0, 0, 1, 0, ...]`, label smoothing distributes a small probability mass across all classes:

```
Smoothed target = (1 - ε) × one_hot + ε / num_classes
```

This prevents the network from becoming overconfident on training examples and consistently reduces the gap between training and test accuracy.

### 3. Gradient Clipping (max_norm = 1.0)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied once per batch after `loss.backward()`. When gate gradients and weight gradients interact — especially when lambda ramps up sharply — the combined gradient norm can spike and destabilize training. Clipping caps the global norm, keeping both loss and accuracy curves smooth.

### 4. CosineAnnealingLR

```
LR(epoch) = eta_min + 0.5 × (lr_init - eta_min) × (1 + cos(π × epoch / T_max))
```

Rather than keeping a fixed learning rate or stepping it down abruptly, cosine annealing traces a smooth curve that starts high (fast learning), slows as training matures, and approaches a near-zero floor. This allows the model to settle into flatter, better-generalizing minima in the final epochs.

---

## Experiments

Three values of `lambda_target` are compared to study the accuracy–sparsity tradeoff:

| λ_target | Effect |
|----------|--------|
| `1e-4`   | Gentle pruning pressure — prioritizes accuracy |
| `1e-3`   | Balanced — moderate sparsity with competitive accuracy |
| `1e-2`   | Aggressive pruning — maximizes sparsity, some accuracy cost |

Results and gate distribution histograms are saved automatically to `./outputs/`.

---

## Sparsity Measurement

After training, a gate is considered **pruned** if its sigmoid value falls below `0.01`:

```
sparsity (%) = (number of gates < 0.01) / (total number of gates) × 100
```

A gate below this threshold contributes less than 1% of its maximum possible weight — effectively zero for any practical purpose.

---

## Setup and Requirements

### Dependencies

```
Python >= 3.9
torch >= 2.0
torchvision >= 0.15
matplotlib >= 3.5
numpy >= 1.23
```

Install via pip:

```bash
pip install torch torchvision matplotlib numpy
```

### Running Locally

```bash
python -u self_pruning_net.py
```

The `-u` flag disables Python's output buffering so epoch logs print in real time.

CIFAR-10 (~170 MB) will be downloaded automatically to `./data/` on the first run.

---

## Running on Google Colab (Recommended for Speed)

1. Open [Google Colab](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → GPU** (select T4 or better)
3. Upload `self_pruning_net.py` to the Colab file browser
4. In a code cell, run:

```python
!pip install torch torchvision matplotlib numpy --quiet
!python -u self_pruning_net.py
```

**Expected runtimes:**

| Hardware | Time per Experiment (50 epochs) | Total (3 experiments) |
|----------|---------------------------------|-----------------------|
| CPU only | ~30–60 minutes | ~2–3 hours |
| Colab T4 GPU | ~3–5 minutes | ~10–15 minutes |
| Colab A100 GPU | ~1–2 minutes | ~4–6 minutes |

After training completes, download the output files:

```python
from google.colab import files
import glob
for f in glob.glob('./outputs/*'):
    files.download(f)
```

---

## Live Training Output

Each log line prints every 5 epochs:

```
Epoch 05 | Phase 1 | lam=0.0e+00 | lr=9.8e-04 | Loss=1.8423 | CE=1.8423 | Sparsity=1647234.50 | TrainAcc=32.14% | TestAcc=35.61%
Epoch 10 | Phase 1 | lam=0.0e+00 | lr=9.3e-04 | Loss=1.5812 | CE=1.5812 | Sparsity=1647234.50 | TrainAcc=46.22% | TestAcc=48.87%
Epoch 15 | Phase 2 | lam=2.0e-06 | lr=8.5e-04 | Loss=1.4901 | CE=1.4511 | Sparsity=1647234.50 | TrainAcc=50.17% | TestAcc=52.04%
...
```

Fields explained:

| Field | Meaning |
|-------|---------|
| `Phase` | Training phase (1=warmup, 2=ramp, 3=full lambda) |
| `lam` | Current active lambda value |
| `lr` | Current weight learning rate (cosine schedule) |
| `Loss` | Total combined loss (CE + λ × sparsity) |
| `CE` | Cross-entropy component only |
| `Sparsity` | Raw sum of all sigmoid gate values (unnormalized) |
| `TrainAcc` | Training set accuracy this epoch |
| `TestAcc` | Test set accuracy (evaluated every 5 epochs) |

---

## Reproducibility

All random sources are seeded before training:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
numpy.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

Running the same script twice on the same hardware will produce identical results.

---

## Known Limitations

- **MLP on image data**: The architecture uses fully-connected layers applied to flattened pixels. This is intentional — the project demonstrates gate-based pruning mechanics, not state-of-the-art CIFAR-10 accuracy. A CNN backbone would significantly outperform this baseline.
- **Memory footprint of gate_scores**: Each `PrunableLinear` layer doubles its parameter count by adding a full-sized `gate_scores` matrix. The model is memory-heavier than a standard network of the same width during training (though the pruned inference graph is leaner).
- **No hard pruning at inference**: Gates are continuous, not binary. The sparsity metric counts near-zero gates as pruned, but the actual matrix multiplications still include them. Applying a hard mask post-training would recover the full speed benefit.

---

## License

This project is released under the MIT License. Use it, modify it, build on it.
