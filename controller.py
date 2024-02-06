import json
import jax.numpy as jnp
from jax import vmap
from collections import deque
import numpy as np

# Load the configuration file
with open('config.json', 'r') as file:
    config = json.load(file)

# Accessing a specific configuration setting

class ClassicController:
    def __init__(self, learning_rate, params):
        self.learning_rate = learning_rate
        self.error_history = []
        self.params = jnp.array(params)
    
    def update_params(self, grads):
        params -= self.learning_rate * grads
        self.error_history = []
        return params

    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        if len(self.error_history) >= 2:
            error_de = current_error - self.error_history[-2]
        else:
            error_de = 0  # Set error_de to 0 if there are fewer than 2 elements in error_history
        error_sum = jnp.sum(jnp.array(self.error_history))
        error_array = jnp.array([current_error, error_sum, error_de])
        return jnp.dot(self.params, error_array)
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    



    

class NeuralNetworkController():
    def __init__(self, learning_rate, activation_functions, params):
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.error_history = []
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in params]

    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        error_sum = jnp.sum(jnp.array(self.error_history))
        if len(self.error_history)>1:
            error_de = current_error - self.error_history[-2]
        else:
            error_de = 0
        error_array = jnp.array([current_error, error_sum, error_de])
        
        return self.predict(error_array)

    def predict(self, features):
        activations = jnp.array(features)
        for i, (weights, biases) in enumerate(self.params):
            # Apply the specified activation function for the current layer
            activation_func = self.get_activation_function(i)
            activations = jnp.dot(activations, jnp.array(weights)) + jnp.array(biases)
            activations = activation_func(activations)
        return jnp.squeeze(activations)
    
    def get_activation_function(self, layer_index):
        # Get the activation function based on the layer index
        if layer_index < len(self.activation_functions):
            activation_func_str = self.activation_functions[layer_index]
            if activation_func_str == "Sigmoid":
                return self.sigmoid
            elif activation_func_str == "Tanh":
                return jnp.tanh
            elif activation_func_str == "RELU":
                return self.relu
            else:
                raise ValueError(f"Unsupported activation function: {activation_func_str}")
        else:
            raise ValueError(f"No activation function specified for layer {layer_index}")
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def relu(x):
        return jnp.maximum(0, x)

    # Sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    

