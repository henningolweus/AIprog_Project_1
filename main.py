import random
from plant import Bathtub, CournotCompetition, LogisticGrowthPlant
from controller import ClassicController, NeuralNetworkController
import jax
import jax.numpy as jnp
import numpy as np
import json
import matplotlib.pyplot as plt
from jax import jit


class ConSys:
    def __init__(self, config):
        self.config = config

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

    def run_one_epoch(self, params, disturbance_array):
        disturbance_array = jnp.array(disturbance_array)
        control_signal = 0
        controller = self.init_controller(params)
        plant = self.init_plant() 
        output_history = []

        for i in range(self.config["simulation"]["timesteps_per_epoch"]):
            output = plant.update_state(control_signal,disturbance_array[i])
            error = self.config["simulation"]["target_value"] - output
            control_signal = controller.compute_control_signal(error)
            output_history.append(output)

        return controller.loss_function(), output_history

    def update_params(self, params, grads, learning_rate, clip_value=1.0):
        # Gradient clipping function
        #calculate the Euclidean norm of the gradient vector
        def clip_gradient(gradient):

            norm = jnp.linalg.norm(gradient)
            return jnp.where(norm > clip_value, gradient * clip_value / norm, gradient)

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
        
    def run_system(self):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0,has_aux=True) #computes both the value and gradient of self.run_one_epoch (which returns MSE from loss function) with respect to its first argument
        params = self.init_params() #initializes the parameters of the controller from the config file
        kp_history, ki_history, kd_history, mse_history = [], [], [], [], 

        for i in range(self.config["simulation"]["num_epochs"]):
            disturbance_array = np.random.uniform(-0.01, 0.01, self.config["simulation"]["timesteps_per_epoch"])
            (mse, output_history), grads = gradfunc(params, disturbance_array) #Calculates the MSE for the current epoch, and the gradients of the MSE with respect to the parameters
            mse_history.append(mse)
            print("GRADS JUST CALCULATED: ")
            print(grads)
            # Update parameters
            learning_rate = self.config["simulation"]["learning_rate"]
            print("OLD AND NEW PARAMS")
            print(params)
            params = self.update_params(params, grads, learning_rate)
            print(params)

            print("MSE")
            print(mse)
            if self.config["controller"]["type"] == "classic":
                kp_history.append(params[0])
                ki_history.append(params[1])
                kd_history.append(params[2])
            print("epoch: " + str(i+1) + "/" + str(self.config["simulation"]["num_epochs"]))
        
        self.plot_history([kp_history, ki_history, kd_history],'PID Parameters over Epochs','Epoch','Parameter Value',['kp', 'ki', 'kd'])
        self.plot_history(mse_history, 'Mean Squared Error over Epochs', 'Epoch', 'MSE Value', ['MSE'])
        #print(error_history)
        
        #self.plot_history(error_history, 'Error History over Timesteps', 'Timestep', 'Error', ['Error per timestep'])
        self.plot_history(output_history, 'Output History over Timesteps', 'Timestep', 'Output', ['Output per timestep'])
        #self.plot_history(Control_history, 'Control History over Timesteps', 'Timestep', 'Control signal', ['Error per timestep'])


    def gen_neural_net_params(self):
        layers = self.config["controller"]["neural_net"]["layers"]
        weight_range = self.config["controller"]["neural_net"]["weight_initial_range"]
        bias_range = self.config["controller"]["neural_net"]["bias_initial_range"]
        layers = self.config["controller"]["neural_net"]["layers"]
        sender = layers[0]; params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(weight_range[0],weight_range[1], (sender, receiver))
            biases = np.random.uniform(bias_range[0], bias_range[1],(1,receiver))
            params.append([weights, biases])
            sender = receiver
        return params
    
    def init_params(self):
        if self.config["controller"]["type"] == "classic":
            params = jnp.array([config["controller"]["classic"]["kp"],config["controller"]["classic"]["ki"], config["controller"]["classic"]["kd"] ])
        elif self.config["controller"]["type"] == "neural_net":
            params = self.gen_neural_net_params()
        else:
            raise ValueError(f"Unsupported controller type: {self.config['controller']['type']}")
        return params
    
    def init_controller(self, params):
        if self.config["controller"]["type"] == "classic":
            controller = ClassicController( config['simulation']['learning_rate'], params)
        elif self.config["controller"]["type"] == "neural_net":
            controller = NeuralNetworkController( config['simulation']['learning_rate'], config['controller']['neural_net']["activation_functions"], params )
        else:
            raise ValueError(f"Unsupported controller type: {self.config['controller']['type']}")
        
        return controller
    
    def init_plant(self):
        if self.config["plant"]["type"] == "bathtub":
            plant = Bathtub(self.config)
        elif self.config["plant"]["type"] == "cournot_competition":
            plant = CournotCompetition(self.config)
        elif self.config["plant"]["type"] == "logistic_growth":
            plant = LogisticGrowthPlant(self.config)
        else:
            raise ValueError(f"Unsupported plant type: {self.config['plant']['type']}")
        return plant
    
    def init_controller_and_plant(self):
        params = self.init_params()
        controller = self.init_controller(params)
        plant = self.init_plant()
        
        return controller, plant

        


            

with open('config.json', 'r') as file:
    config = json.load(file)



consys = ConSys(config)
consys.run_system()
