# EVA4 Session 5 - CNN Implementation & Concepts

This repository contains the implementation and theoretical concepts from EVA4 Session 5, focusing on Convolutional Neural Networks (CNNs) for MNIST digit classification.

## 📁 Project Structure

```
era4-assign5/
├── EVA4_Session_5.ipynb    # Main implementation notebook
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.6+
- PyTorch
- torchvision
- torchsummary
- tqdm

### Installation
```bash
pip install torch torchvision torchsummary tqdm
```

### Running the Code
1. Open `EVA4_Session_5.ipynb` in Jupyter Notebook
2. Run all cells to train the CNN model on MNIST dataset
3. The model will automatically download MNIST data and train for 1 epoch

## 🧠 CNN Architecture

The implemented CNN follows this architecture:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)    # 1→32 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)   # 32→64 channels
        self.pool1 = nn.MaxPool2d(2, 2)                # 2x2 max pooling
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64→128 channels
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 128→256 channels
        self.pool2 = nn.MaxPool2d(2, 2)                # 2x2 max pooling
        self.conv5 = nn.Conv2d(256, 512, 3)            # 256→512 channels
        self.conv6 = nn.Conv2d(512, 1024, 3)           # 512→1024 channels
        self.conv7 = nn.Conv2d(1024, 10, 3)            # 1024→10 channels (classes)
```

### Architecture Details
- **Input**: 28×28×1 (MNIST grayscale images)
- **Output**: 10 classes (digits 0-9)
- **Total Parameters**: ~6.2M parameters
- **Activation**: ReLU + Log Softmax
- **Optimizer**: SGD with momentum (lr=0.01, momentum=0.9)

## 📊 Model Summary

The model uses `torchsummary` to display architecture details:
- Input size: (1, 28, 28)
- Output size: (10,)
- Total parameters and trainable parameters are displayed

## 🎯 Training Configuration

- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Batch Size**: 128
- **Epochs**: 1 (for demonstration)
- **Data Augmentation**: Normalization (mean=0.1307, std=0.3081)
- **Loss Function**: Negative Log Likelihood Loss
- **Device**: Automatic CUDA detection

## 📈 Key Features

1. **GPU Support**: Automatic CUDA detection and device placement
2. **Progress Tracking**: tqdm progress bars during training
3. **Model Summary**: Detailed architecture visualization
4. **Data Loading**: Efficient DataLoader with proper normalization
5. **Training Loop**: Complete train/test cycle implementation

## 🧮 CNN Theory & Calculations

### 1. Convolution Output Size
For a 3×3 kernel on a 47×49 image:
```
Output size = Input size - Kernel size + 1
Height: 47 - 3 + 1 = 45
Width: 49 - 3 + 1 = 47
```

### 2. Receptive Field Calculation
To reach a 21×21 receptive field with 3×3 kernels:
```
R = 1 + n × (k-1), where k=3
21 = 1 + n × 2 → n = 10 layers
```

### 3. Parameter Calculation
For 49×49×256 input with 512 kernels of size 3×3:
```
Parameters per kernel = 256 × 3 × 3 = 2,304
Total parameters = 2,304 × 512 = 1,179,648
```

### 4. CNN Design Principles
- ✅ Use padding to maintain spatial dimensions
- ✅ Prefer stride=1 unless pooling is applied
- ✅ Add layers to reach full receptive field
- ✅ 3×3 kernels are common but not mandatory

### 5. Max-Pooling Benefits
- ✅ Reduces spatial dimensions (H×W)
- ✅ Provides translational invariance
- ❌ Does NOT reduce channel count
- ❌ Does NOT provide rotational invariance

## 🔧 Usage Examples

### Training the Model
```python
# Initialize model and optimizer
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

### Model Summary
```python
from torchsummary import summary
summary(model, input_size=(1, 28, 28))
```

## 📚 Learning Objectives

This session covers:
1. **CNN Architecture Design**: Layer selection and parameter tuning
2. **Receptive Field Calculations**: Understanding feature map growth
3. **Parameter Counting**: Memory and computational considerations
4. **Training Implementation**: Complete PyTorch training pipeline
5. **GPU Utilization**: CUDA integration and optimization

## 🎓 Key Takeaways

- **Receptive field** grows by (k-1) per convolution layer
- **Kernel parameters** = (input_channels × kernel_size²) × num_kernels
- **Max-pooling** reduces spatial size while preserving important features
- **Information retention** through convolution and pooling is about re-encoding, not discarding
- **GPU memory** usage depends on feature map channels × spatial dimensions

## 📖 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CNN Visualization](https://github.com/utkuozbulak/pytorch-cnn-visualizations)

---

**Note**: This implementation is part of the EVA4 (Extreme Vision AI) course curriculum, focusing on practical deep learning applications and theoretical understanding of CNNs.