# XOR Neural Network from Scratch
This repository implements a **feedforward neural network** to solve the XOR logic problem using just Python.

## How It Works
The network learns the XOR operation through supervised training using **backpropagation**. It features a three-layer architecture (input, hidden, output) using the sigmoid activation function and gradient descent optimization.

## Key Features
* **Pure Python Implementation**: Built using only Python's standard library (`random` and `math`)
* **Three-Layer Architecture**:
  * 2 input neurons
  * 3 hidden neurons
  * 1 output neuron
* **Complete Neural Network Components**:
  * Forward propagation
  * Backward propagation
  * Weight updates
  * Sigmoid activation
  * Gradient descent optimization

## File Overview
* **xor_nn.py**:
  * `initialize_weights()`: Creates random initial weights for hidden and output layers
  * `sigmoid()`: Implements the sigmoid activation function
  * `predict()`: Performs forward propagation to generate predictions
  * `train()`: Implements the backpropagation algorithm
  * `run_epoch()`: Processes one complete pass through the training data
  * `train_network()`: Manages the complete training process

## Neural Network Components
### Network Architecture
```
Input Layer (2 neurons)
     ↓
Hidden Layer (3 neurons)
     ↓
Output Layer (1 neuron)
```

### Key Functions
* **Initialization**:
  * Random weights in range [-1, 1]
  * Separate weight matrices for hidden and output layers
  * Includes bias terms

* **Forward Propagation**:
  * Computes hidden layer activations
  * Applies sigmoid activation function
  * Calculates final output

* **Backward Propagation**:
  * Calculates output error
  * Computes hidden layer error
  * Updates weights using gradient descent

## Example Output
```python
Epoch 1
Input: [0, 0], Predicted Output: 0.5123
Input: [0, 1], Predicted Output: 0.4891
Input: [1, 0], Predicted Output: 0.5201
Input: [1, 1], Predicted Output: 0.4799

...

Epoch 20000
Input: [0, 0], Predicted Output: 0.0521
Input: [0, 1], Predicted Output: 0.9479
Input: [1, 0], Predicted Output: 0.9481
Input: [1, 1], Predicted Output: 0.0519
```

## How to Use
1. Clone the repository
2. Run the XOR network:
```bash
python xor_nn.py
```
3. The program will:
   * Initialize the network
   * Train for 20,000 epochs
   * Display predictions after each epoch
   * Show final weights

## Training Parameters
* Learning Rate: 0.1
* Number of Epochs: 20,000
* Training Data:
  * [0,0] → 0
  * [0,1] → 1
  * [1,0] → 1
  * [1,1] → 0

## Future Improvements
* Add **batch training** capability
* Implement different **activation functions** (ReLU, tanh)
* Add **momentum** to gradient descent
* Include **regularization** techniques
* Add **visualization** of:
  * Training progress
  * Decision boundaries
  * Network architecture
