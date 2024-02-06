from plant import Bathtub, CournotCompetition, LogisticGrowthPlant
from controller import ClassicController, NeuralNetworkController
import jax
import jax.numpy as jnp
import numpy as np
import json
import matplotlib.pyplot as plt


class ConSys:
    def __init__(self, config):
        # Initialize control system with configuration settings
        self.config = config

    def run_one_epoch(self, params, disturbance_array):
        # Simulate one epoch (cycle) of the control system
        disturbance_array = jnp.array(disturbance_array)
        control_signal = 0
        controller = self.init_controller(params) # Initialize controller with parameters
        plant = self.init_plant() 

        # Loop through each timestep in the epoch
        for i in range(self.config["simulation"]["timesteps_per_epoch"]):
            # Update plant state and compute error
            output = plant.update_state(control_signal,disturbance_array[i])
            error = self.config["simulation"]["target_value"] - output
            # Compute new control signal based on error
            control_signal = controller.compute_control_signal(error)
        
        # Return MSE for the epoch
        return controller.loss_function()
        
    def run_system(self):
        # Main loop to run the control system simulation
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0) # Setup for automatic differentiation
        params = self.init_params() # Initialize parameters based on config
        kp_history, ki_history, kd_history, mse_history = [], [], [], [], 
        learning_rate = self.config["simulation"]["learning_rate"]

        # Run simulation for configured number of epochs
        for i in range(self.config["simulation"]["num_epochs"]):
            # Generate disturbance array for the epoch
            disturbance_array = np.random.uniform(-0.01, 0.01, self.config["simulation"]["timesteps_per_epoch"])
            # Compute MSE and gradients for current epoch
            mse, grads = gradfunc(params, disturbance_array) 
            mse_history.append(mse)
            # Update parameters
            params = self.update_params(params, grads, learning_rate)
            # Track PID parameters if using classic controller
            if self.config["controller"]["type"] == "classic":
                kp_history.append(params[0])
                ki_history.append(params[1])
                kd_history.append(params[2])
            # Print progress
            print("MSE")
            print(mse)
            print("epoch: " + str(i+1) + "/" + str(self.config["simulation"]["num_epochs"]))
        
        # Plotting histories for analysis
        self.plot_history([kp_history, ki_history, kd_history],'PID Parameters over Epochs','Epoch','Parameter Value',['kp', 'ki', 'kd'])
        self.plot_history(mse_history, 'Mean Squared Error over Epochs', 'Epoch', 'MSE Value', ['MSE'])

    def update_params(self, params, grads, learning_rate, clip_value=1.0):
        # Update parameters with gradient descent and gradient clipping
        # Gradient clipping wqs introduced to deal with reocurring NaN-outputs from gradfunc
        def clip_gradient(gradient):
            norm = jnp.linalg.norm(gradient)
            return jnp.where(norm > clip_value, gradient * clip_value / norm, gradient)
        
        # Update for neural network or classical controller parameters
        if isinstance(params, list) and isinstance(params[0], list):  # Check if params is a list of lists (neural network)
            #Check if params is a list of lists (neural network)
            updated_params = []
            for (w, b), (gw, gb) in zip(params, grads):
                # Clip gradients
                clipped_gw = clip_gradient(gw)
                clipped_gb = clip_gradient(gb)

                # Update weights and biases with clipped gradients
                new_w = w - learning_rate * clipped_gw
                new_b = b - learning_rate * clipped_gb
                updated_params.append([new_w, new_b])
            return updated_params
        else:  # Assuming params is a simple list/array (classical PID)
            # Clip gradients if it's a single array
            clipped_grads = clip_gradient(grads)
            return params - learning_rate * clipped_grads

    def gen_neural_net_params(self):
        # Generate initial parameters for neural network controller

        layers = self.config["controller"]["neural_net"]["layers"]
        weight_range = self.config["controller"]["neural_net"]["weight_initial_range"]
        bias_range = self.config["controller"]["neural_net"]["bias_initial_range"]
        layers = self.config["controller"]["neural_net"]["layers"]
        sender = layers[0]; params = []
        # Initialize weights and biases for each layer  
        for receiver in layers[1:]:
            # Initialize weights and biases for the current layer with the shape specified in the config file
            weights = np.random.uniform(weight_range[0],weight_range[1], (sender, receiver))
            biases = np.random.uniform(bias_range[0], bias_range[1],(1,receiver))
            # Append the weights and biases for the current layer to the parameters list
            params.append([weights, biases])
            # Update the sender size for the next layer's connections
            sender = receiver
        # Return the complete list of parameters (weights and biases) for all layers
        return params
    
    def init_params(self):
        # Initialize parameters, check if the controller is "classic" or "neural_net"
        if self.config["controller"]["type"] == "classic":
            params = jnp.array([config["controller"]["classic"]["kp"],config["controller"]["classic"]["ki"], config["controller"]["classic"]["kd"] ])
        elif self.config["controller"]["type"] == "neural_net":
            params = self.gen_neural_net_params()
        else:
            raise ValueError(f"Unsupported controller type: {self.config['controller']['type']}")
        return params
    
    def init_controller(self, params):
        # Initialize controller with the given params, check if the controller is classic or neural_net
        if self.config["controller"]["type"] == "classic":
            controller = ClassicController( config['simulation']['learning_rate'], params)
        elif self.config["controller"]["type"] == "neural_net":
            controller = NeuralNetworkController( config['simulation']['learning_rate'], config['controller']['neural_net']["activation_functions"], params )
        else:
            raise ValueError(f"Unsupported controller type: {self.config['controller']['type']}")
        
        return controller
    
    def init_plant(self):
        # Initialize the plant based on the specified plant in the config file
        if self.config["plant"]["type"] == "bathtub":
            plant = Bathtub(self.config)
        elif self.config["plant"]["type"] == "cournot_competition":
            plant = CournotCompetition(self.config)
        elif self.config["plant"]["type"] == "logistic_growth":
            plant = LogisticGrowthPlant(self.config)
        else:
            raise ValueError(f"Unsupported plant type: {self.config['plant']['type']}")
        return plant

    def plot_history(self, series_list, title, x_label, y_label, legend_labels):
        plt.figure(figsize=(12, 5))
        # Check if it's a single series or multiple series
        if isinstance(series_list[0], list):
            # Plotting each series in the list
            for series, label in zip(series_list, legend_labels):
                plt.plot(series, label=label)
        else:
            # Plotting a single series
            plt.plot(series_list, label=legend_labels[0])
        # Setting up the plot
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True)
        plt.show()
    

with open('config.json', 'r') as file:
    config = json.load(file)

consys = ConSys(config)
consys.run_system()
