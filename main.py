import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'Plant_growth.csv'
data = pd.read_csv(data_path)
print(data.head())

# Handle missing values (if any)
data = data.dropna()

# Assuming 'Growth_Milestone' is the target variable
X = data.drop('Growth_Milestone', axis=1)
y = data['Growth_Milestone']

# Encode categorical variables
X = pd.get_dummies(X)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode target variable if it is categorical
label_encoder = LabelEncoder()	
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

# Initialize weights and biases
def initialize_weights(layers):
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
        biases.append(np.zeros((1, layers[i+1])))
    return weights, biases

# Forward propagation
def forward_propagation(X, weights, biases):
    activations = [X]
    z_values = []
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        z_values.append(z)
        if i == len(weights) - 1:
            activation = sigmoid(z)
        else:
            activation = relu(z)
        activations.append(activation)
    return activations, z_values

# Compute cost
def compute_cost(y_true, y_pred):
    m = y_true.shape[0]
    cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cost

# Backward propagation
def backward_propagation(y_true, activations, z_values, weights):
    m = y_true.shape[0]
    dz = activations[-1] - y_true
    dw = (1/m) * np.dot(activations[-2].T, dz)
    db = (1/m) * np.sum(dz, axis=0, keepdims=True)
    grads = [(dw, db)]
    
    for i in range(len(weights) - 2, -1, -1):
        dz = np.dot(dz, weights[i + 1].T) * relu_derivative(z_values[i])
        dw = (1/m) * np.dot(activations[i].T, dz)
        db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        grads.append((dw, db))
    
    grads.reverse()
    return grads

# Update weights and biases
def update_parameters(weights, biases, grads, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * grads[i][0]
        biases[i] -= learning_rate * grads[i][1]
    return weights, biases

# Training the neural network
def train_neural_network(x_train, y_train, layers, learning_rate, epochs):
    weights, biases = initialize_weights(layers)
    costs = []
    for epoch in range(epochs):
        activations, z_values = forward_propagation(x_train, weights, biases)
        cost = compute_cost(y_train, activations[-1])
        costs.append(cost)
        grads = backward_propagation(y_train, activations, z_values, weights)
        weights, biases = update_parameters(weights, biases, grads, learning_rate)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Cost: {cost}')
    return weights, biases, costs

# Predict function
def predict(X, weights, biases):
    activations, _ = forward_propagation(X, weights, biases)
    return np.round(activations[-1])

# Define the architecture
layers = [x_train.shape[1], 64, 32, 16, 1]  # Input layer, hidden layers, output layer

# Train the model
weights, biases, costs = train_neural_network(x_train, y_train.reshape(-1, 1), layers, learning_rate=0.01, epochs=1000)

# Make predictions
y_pred_train = predict(x_train, weights, biases)
y_pred_test = predict(x_test, weights, biases)

# Calculate accuracy
train_accuracy = np.mean(y_pred_train == y_train.reshape(-1, 1)) * 100
test_accuracy = np.mean(y_pred_test == y_test.reshape(-1, 1)) * 100

print(f'Training Accuracy: {train_accuracy}%')
print(f'Testing Accuracy: {test_accuracy}%')

# Plot the cost over epochs
plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost reduction over epochs')
plt.show()