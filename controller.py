import json
import jax.numpy as jnp
from jax import vmap
import numpy as np

# Load the configuration file
with open('config.json', 'r') as file:
    config = json.load(file)

# Accessing a specific configuration setting

class ClassicController:
    def __init__(self, params, learning_rate):
        self.params = params
        self.learning_rate = learning_rate
        self.error_history = []

    
    def update_params(self, grads):
        self.params -= self.learning_rate * grads
        return self.params
    
    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        error_sum = jnp.sum(jnp.array(self.error_history))
        error_de = current_error - self.error_history[-1]

        error_array = jnp.array([current_error, error_sum, error_de])
        return jnp.dot(self.params, error_array)
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def get_error_history(self):
        return self.error_history




    

class NeuralNetworkController():
    def __init__(self, params, learning_rate, activation_func):
        self.params = params
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.error_history = []
        # Initialize weights and biases for each layer based on the 'layers' configuration

    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        error_sum = jnp.sum(jnp.array(self.error_history))
        error_de = current_error - self.error_history[-1]
        error_array = jnp.array([current_error, error_sum, error_de])
        
        return self.predict(self.params, error_array)[0][0]
    
    def get_error_history(self):
        return self.error_history

    
    def predict(self, params, features):
        if self.activation_func == "Sigmoid":
            activation_func = self.sigmoid
        elif self.activation_func == "Tanh":
            activation_func = jnp.tanh
        else:
            activation_func = self.relu
        
        activations = features
        for weights, biases in params:
            activations = jnp.dot(activations, weights) + biases
            activations = activation_func(activations)
        return activations
            
    
    def model(self,params, cases):
        return jnp.array([jnp.dot(params[0:-1],case) + params[-1] for case in cases])
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def update_not_in_use(self, loss_gradient):
        self.params-= self.learning_rate*loss_gradient()
        return self.params

    def update_params(self, grads):
        self.params-= self.learning_rate*grads
        return self.params
        # ReLU activation function
    def relu(x):
        return jnp.maximum(0, x)

    # Sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))