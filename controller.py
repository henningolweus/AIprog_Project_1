import json
import jax.numpy as jnp
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
    
    def compute_control_signal(self, current_error):
        self.error_history.append(current_error)
        self.error_sum = jnp.sum(jnp.array(self.error_history))
        error_de = current_error - self.error_history[-1]

        error_array = jnp.array([current_error, self.error_sum, error_de])
        return jnp.dot(self.params, error_array)
    
    def loss_function(self):
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def get_error_history(self):
        return self.error_history




    

class NeuralNetworkController():
    def __init__(self, config):
        self.layers = config['controller']['neural_net']['neurons_per_layer']
        # Initialize weights and biases for each layer based on the 'layers' configuration

    def update(self, input_vector):
        # Implement the neural network inference logic here
        # For now, let's just return a dummy control signal
        control_signal = np.random.uniform(-1, 1)
        return control_signal