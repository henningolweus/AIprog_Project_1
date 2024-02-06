import json
import jax.numpy as jnp


class ClassicController:
    def __init__(self, learning_rate, params):
        # Initialize controller with learning rate and parameters
        self.learning_rate = learning_rate
        self.error_history = []
        self.params = jnp.array(params)

    def compute_control_signal(self, current_error):
        # Compute control signal based on current error
        self.error_history.append(current_error)
        if len(self.error_history) > 1:
            # Compute derivative of error if history is long enough
            error_de = current_error - self.error_history[-2]
        else:
            error_de = 0  # Set error_de to 0 if there are fewer than 2 elements in error_history
        error_sum = jnp.sum(jnp.array(self.error_history))  # Sum of errors for integral component
        error_array = jnp.array([current_error, error_sum, error_de])  # Error components array
        return jnp.dot(self.params, error_array)  # Calculate control signal
    
    def loss_function(self):
        # Calculate mean squared error (MSE) of error history
        return jnp.mean(jnp.array(self.error_history)**2)
    



    

class NeuralNetworkController():
    def __init__(self, learning_rate, activation_functions, params):
        # Initialize NN controller with learning rate, activation functions, and parameters
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.error_history = []
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in params] # Convert params to JAX numpy array for tracing

    def compute_control_signal(self, current_error):
        # Compute control signal based on current error
        self.error_history.append(current_error)
        error_sum = jnp.sum(jnp.array(self.error_history))
        if len(self.error_history)>1:
            error_de = current_error - self.error_history[-2]
        else:
            error_de = 0
        # Array of current error, sum of errors, and error difference
        error_array = jnp.array([current_error, error_sum, error_de])
        
        return self.predict(error_array)

    def predict(self, features):
        # Predict output based on input features using the neural network
        activations = jnp.array(features)
        for i, (weights, biases) in enumerate(self.params):
            # Apply the specified activation function for the current layer
            activation_func = self.get_activation_function(i)
            activations = jnp.dot(activations, jnp.array(weights)) + jnp.array(biases)
            activations = activation_func(activations)
        return jnp.squeeze(activations) # Remove extra dimensions
    
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
        # Compute mean squared error of error history as loss
        return jnp.mean(jnp.array(self.error_history)**2)
    
    def relu(x):
        # Rectified Linear Unit activation function
        return jnp.maximum(0, x)

    # Sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + jnp.exp(-x))
    

