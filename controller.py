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
    
    def get_error_history(self):
        return self.error_history



    

class NeuralNetworkController():
    def __init__(self, learning_rate, activation_functions, params):
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.error_history = []
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in params]
        # Initialize weights and biases for each layer based on the 'layers' configuration

    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        error_sum = jnp.sum(jnp.array(self.error_history))
        if len(self.error_history)>1:
            error_de = current_error - self.error_history[-2]
        else:
            error_de = 0
        error_array = jnp.array([current_error, error_sum, error_de])
        
        return self.predict(error_array)
    
    def get_error_history(self):
        return self.error_history


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
    
    

    def model(self,params, cases):
        return jnp.array([jnp.dot(params[0:-1],case) + params[-1] for case in cases])
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    
    def relu(x):
        return jnp.maximum(0, x)

    # Sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    

    # def update_params(self, grads):
    #     print("######PARAMS######")
    #     params = [(jnp.array(w), jnp.array(b)) for w, b in params]
    #     print(params)
    #     print("######GRADS######")
    #     print(grads)
    #     new_params = []

    #     for (w, b), (gw, gb) in zip(params, grads):
    #         new_w = jnp.array(w) - self.learning_rate * jnp.array(gw)
    #         new_b = jnp.array(b) - self.learning_rate * jnp.array(gb)
    #         new_params.append([new_w, new_b])

    #     params = new_params
    #     self.error_history = []
    #     return params
    #     # ReLU activation function


    # def update_not_in_use(self, loss_gradient):

    #     self.params-= self.learning_rate*loss_gradient()
    #     return self.params