# Experiment Report

## Title: Effect of Activation Functions on Neural Network Convergence and Efficiency for MNIST Dataset

### Abstract
This experiment was designed to assess how different activation functions (ReLU, Sigmoid, Tanh) influence the convergence rate and efficiency of a neural network when trained on the MNIST dataset. Using a feedforward neural network with two hidden layers, various activation functions were tested to evaluate their impact on training loss, test accuracy, and computation time. The results indicated that ReLU offered superior accuracy, though none of the configurations achieved the desired training loss threshold within the specified epochs.

### Introduction
#### Research Question
How do different activation functions affect convergence rate and efficiency in neural networks applied to the MNIST dataset?

#### Hypothesis
The choice of activation function (ReLU, Sigmoid, Tanh) affects the convergence rate and efficiency of a neural network on the MNIST dataset.

#### Background
Neural network performance can be significantly impacted by the choice of activation function. ReLU, Sigmoid, and Tanh are commonly used activations, each with unique characteristics affecting learning dynamics. This study seeks to understand their effects on convergence and training efficiency.

### Methodology
#### Experiment Design
A feedforward neural network was used, consisting of an input layer, two hidden layers with 128 neurons each, and an output layer classifying MNIST digits.

#### Experimental Setup
- **Architecture**: Feedforward neural network
  - Input Layer: 784 neurons
  - Hidden Layers: 2 layers, 128 neurons each
  - Output Layer: 10 neurons
- **Activation Functions Tested**: ReLU, Sigmoid, Tanh
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum (0.9)
- **Loss Function**: Cross-entropy loss
- **Weight Initialization**: Xavier initialization
- **Batch Size**: 64
- **Learning Rate**: 0.01
- **Epochs**: 50 (with early stopping when loss < 0.05 for 5 consecutive epochs)
- **Random Seed**: 42

#### Execution
The experiments were executed on the MNIST dataset using PyTorch, tracking metrics such as training loss over epochs, test accuracy post-training, epochs to achieve desired training loss (0.1), and computation time per epoch.

### Results
#### Control Group (ReLU Activation)
- **Test Accuracy**: 97.80%
- **Epochs to reach loss of 0.1**: Not achieved
- **Computation Time per Epoch**: ~8.3 seconds

#### Experimental Group
- **Sigmoid Activation**
  - **Test Accuracy**: 93.92%
  - **Epochs to reach loss of 0.1**: Not achieved
  - **Computation Time per Epoch**: ~8.18 seconds

- **Tanh Activation**
  - **Test Accuracy**: 97.61%
  - **Epochs to reach loss of 0.1**: Not achieved
  - **Computation Time per Epoch**: ~8.10 seconds

### Conclusion and Future Work
#### Summary of Findings
ReLU function demonstrated the highest test accuracy, performing better than both Sigmoid and Tanh. No activation function enabled the network to achieve the desired training loss threshold within the set epochs.

#### Recommendations for Future Experiments
1. Experiment with different learning rates and optimization techniques.
2. Increase model complexity with additional neurons or layers.
3. Allow longer training (increase epochs) to further assess convergence.
4. Test alternative activation functions such as Swish or Leaky ReLU.

### Appendices
- **Raw Log Excerpts**: Available in '/workspace/research_8f58f396-959e-4d1b-a146-219fe9bb5300/results' directory.
- **Configuration Details**: Scripts and outputs stored with specified naming conventions.
- **Code Commit Hash**: [Not specified in log] 
- **File Paths**: Stored under '/workspace/research_8f58f396-959e-4d1b-a146-219fe9bb5300/'

This report documents the process and findings of the experiment, with suggestions laid out for enhancing future studies based on observed outcomes.