import random
import math

# Define network parameters
H = 3  
N = 2  
K = 1  

# Function to initialize weights
def initialize_weights(H, N, K):
    hidden_layer_weights = [[random.uniform(-1, 1) for _ in range(N + 1)] for _ in range(H)]
    output_layer_weights = [[random.uniform(-1, 1) for _ in range(H + 1)] for _ in range(K)]
    return hidden_layer_weights, output_layer_weights


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def predict(hidden_weights, output_weights, inputs):
  # Forward pass through the hidden layer
  hidden_outputs = []
  for weights in hidden_weights:
    # Compute the output of each hidden node
    activation = weights[0]  # bias
    for i, input_val in enumerate(inputs):
        activation += weights[i + 1] * input_val
    hidden_outputs.append(sigmoid(activation))

# Forward pass through the output layer
  final_outputs = []
  for weights in output_weights:
    # Compute the output of each output node
    activation = weights[0]  # bias
    for i, hidden_output in enumerate(hidden_outputs):
        activation += weights[i + 1] * hidden_output
    final_outputs.append(sigmoid(activation))

  return final_outputs

def train(hidden_weights, output_weights, inputs, targets, learning_rate):
  # Forward pass
  # Compute hidden layer outputs
  hidden_inputs = []
  hidden_outputs = []
  for weights in hidden_weights:
      activation = weights[0]  # Start with the bias
      for i, input_val in enumerate(inputs):
          activation += weights[i + 1] * input_val
      hidden_inputs.append(activation)
      hidden_outputs.append(sigmoid(activation))

  # Compute output layer outputs
  output_inputs = []
  output_outputs = []
  for weights in output_weights:
      activation = weights[0]  # Start with the bias
      for i, hidden_output in enumerate(hidden_outputs):
          activation += weights[i + 1] * hidden_output
      output_inputs.append(activation)
      output_outputs.append(sigmoid(activation))

  # Backward pass
  # Calculate output layer error
  output_errors = [target - output for target, output in zip(targets, output_outputs)]

  # Calculate gradient and update weights for the output layer
  for i, weights in enumerate(output_weights):
      for j in range(len(weights)):
          if j == 0:
              input_val = 1  # Bias input
          else:
              input_val = hidden_outputs[j - 1]
          gradient = output_errors[i] * output_outputs[i] * (1 - output_outputs[i]) * input_val
          weights[j] += learning_rate * gradient

  # Calculate hidden layer error
  hidden_errors = [0] * len(hidden_weights)
  for i in range(len(hidden_weights)):
      error = 0
      for j, weights in enumerate(output_weights):
          error += weights[i + 1] * output_errors[j]
      hidden_errors[i] = error * hidden_outputs[i] * (1 - hidden_outputs[i])

  # Update weights for the hidden layer
  for i, weights in enumerate(hidden_weights):
      for j in range(len(weights)):
          if j == 0:
              input_val = 1  # Bias input
          else:
              input_val = inputs[j - 1]
          gradient = hidden_errors[i] * input_val
          weights[j] += learning_rate * gradient

  return hidden_weights, output_weights

def run_epoch(hidden_weights, output_weights, training_data, learning_rate):

  for inputs, target in training_data:
      hidden_weights, output_weights = train(hidden_weights, output_weights, inputs, [target], learning_rate)
  return hidden_weights, output_weights

def train_network(hidden_weights, output_weights, training_data, learning_rate, num_epochs):

  for epoch in range(num_epochs):
    hidden_weights, output_weights = run_epoch(hidden_weights, output_weights, training_data, learning_rate)

    # Print output predictions after each epoch to observe learning progress
    print(f"Epoch {epoch + 1}")
    for inputs, _ in training_data:
        output = predict(hidden_weights, output_weights, inputs)
        print(f"Input: {inputs}, Predicted Output: {output[0]:.4f}")

  return hidden_weights, output_weights

# Initialize weights right after defining H, N, K and the initialization function
hidden_layer_weights, output_layer_weights = initialize_weights(H, N, K)

# Define the XOR training data
training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Training parameters
learning_rate = 0.1
num_epochs = 20000

# Start training
final_hidden_weights, final_output_weights = train_network(
    hidden_layer_weights, output_layer_weights, training_data, learning_rate, num_epochs
)

# Optionally, print the final weights and check the predictions
print("Final hidden weights:", final_hidden_weights)
print("Final output weights:", final_output_weights)
