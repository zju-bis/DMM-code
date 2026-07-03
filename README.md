# 🧠 DMM: Dual Memory Management for Continual Learning

<p align="center">
  <a href="https://github.com/yourusername/DMM/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.12+-EE4C2C.svg" alt="PyTorch">
  </a>
</p>

> A high-performance PyTorch framework for SNN-based continual training, featuring dynamic memory allocation and optimized experience replay to mitigate catastrophic forgetting.

---

; ## 📖 Introduction

; In the context of continual learning, spiking neural networks often suffer from catastrophic forgetting when sequentially trained on new tasks. **DMM (Dual Memory Management)** provides a robust, memory-efficient solution by intelligently managing replay buffer indices and dynamically swapping memory chunks during continual training phases. 

; The framework introduces a custom `swap_loss` logic, optimizing the trade-off between retaining past task knowledge and acquiring new features. 

; ### ✨ Core Features

; - **Efficient Experience Replay**: Advanced indexing techniques for the Replay Buffer, minimizing memory fragmentation during continuous data streams.
; - **Dynamic Swap Logic**: Implements a novel `swap_loss` objective (e.g., $L_{total} = L_{task} + \alpha L_{swap}$) to dynamically balance the replay of historical data and current task gradients.
; - **Continual Training Ready**: Native support for seamless state transitioning across multiple tasks without requiring full network retraining.
; - **PyTorch Native**: Fully compatible with standard PyTorch training loops and `DataLoader` mechanisms.

; ---
## 🛠️ Prerequisites & Installation

### System Requirements
- Python >= 3.8

### Quick Install

```bash
1. Clone the repository
git clone [https://github.com/yourusername/ProjectName.git](https://github.com/yourusername/ProjectName.git)
cd ProjectName

2. Create and activate a virtual environment (Recommended)
conda create -n DMM python=3.8
conda activate DMM

3. Install dependencies
pip install -r requirements.txt
```
## 🏃 Run Experiments

Our framework is highly modularized. All experiment settings, including hyperparameters and baseline methods, are defined as JSON files in the `exps/` directory.

### 1. Run the Proposed DMM Method
To train the model using our Dual Memory Management (DMM) approach, use the `main.py` script and pass the DMM configuration file along with your target dataset:

```bash
# Run DMM
python main.py --config=./exps/DMM.json

# Run Miro
python main.py --config=./exps/Miro.json

# Run CarM
python main.py --config=./exps/CarM.json

# Run Best Static
python main.py --config=./exps/Best_Static.json

# Run Best History
python main.py --config=./exps/Best_History.json

# Run Heuristic
python main.py --config=./exps/Heuristic.json
```
## 📂 File Structure

```text
DMM/
├── data/                   # Datasets (Caltech101, cifar100, DVS128Gesture, imagenet100, UrbanSound8K, etc.)
├── exps/                   # Experiment configurations in JSON (configs for DMM, carm, miro, etc.)
├── logs/                   # Training logs and outputs
├── models/                 # Directory for different CL models
├── nets/                   # Spiking neural network architectures
│   ├── layer.py            # Custom layers definitions
│   ├── model_setting.py    # Model hyperparameters and basic settings
│   ├── preact_resnet.py    # PreAct-ResNet implementation
│   ├── resnet.py           # Standard ResNet implementation
│   ├── sew_resnet.py       # SEW-ResNet implementation 
│   └── vgg.py              # VGG implementation
├── utils/                  # Utility functions (data loaders, metrics, etc.)
├── main.py                 # Main entry point for training and evaluation
├── README                  # Project documentation
├── requirements.txt        # Python dependencies
└── trainer.py              # Core training loop and continual learning logic
