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
        error_sum = jnp.sum(jnp.array(self.error_history))
        error_de = current_error - self.error_history[-2]
        error_array = jnp.array([current_error, error_sum, error_de])
        return jnp.dot(self.params, error_array)
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def get_error_history(self):
        return self.error_history




    

class NeuralNetworkController():
    def __init__(self, learning_rate, activation_func, params):
        self.activation_func = activation_func
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
        if self.activation_func == "Sigmoid":
            activation_func = self.sigmoid
        elif self.activation_func == "Tanh":
            activation_func = jnp.tanh
        else:
            activation_func = self.relu
        activations = jnp.array(features)
        for weights, biases in self.params:
            activations = jnp.dot(activations, jnp.array(weights)) + jnp.array(biases)
            activations = activation_func(activations)
        return jnp.squeeze(activations)
            
    
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