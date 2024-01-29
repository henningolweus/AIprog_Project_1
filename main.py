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
        control_signal_history = []
        output_history = []

        for i in range(self.config["simulation"]["timesteps_per_epoch"]):
            output = plant.update_state(control_signal,disturbance_array[i])
            error = self.config["simulation"]["target_value"] - output
            control_signal = controller.compute_control_signal(error)
            control_signal_history.append(control_signal)
            output_history.append(output)

        return controller.loss_function(), output_history

    def update_params(self, params, grads, learning_rate):
        if isinstance(params, list) and isinstance(params[0], list):  # Check if params is a list of lists (neural network)
            updated_params = []
            for (w, b), (gw, gb) in zip(params, grads):
                new_w = w - learning_rate * gw
                new_b = b - learning_rate * gb
                updated_params.append([new_w, new_b])
            return updated_params
        else:  # Assuming params is a simple list/array (classical PID)
            return params - learning_rate * grads
        
    def run_system(self):
        gradfunc = jax.value_and_grad(self.run_one_epoch, argnums=0,has_aux=True)
        params = self.init_params()
        
        kp_history, ki_history, kd_history, mse_history = [], [], [], []

        for i in range(self.config["simulation"]["num_epochs"]):
            disturbance_array = np.random.uniform(-0.01, 0.01, self.config["simulation"]["timesteps_per_epoch"])
            (mse, error_history), grads = gradfunc(params, disturbance_array)
            mse_history.append(mse)
            print("GRADS JUST CALCULATED: ")
            print(grads)
            error_history = np.squeeze(error_history)

            # Update parameters
            learning_rate = self.config["simulation"]["learning_rate"]
            print("OLD AND NEW PARAMS")
            print(params)
            params = self.update_params(params, grads, learning_rate)
            print(params)

            print("MSE")
            print(mse)
            
            # kp_history.append(params[0])
            # ki_history.append(params[1])
            # kd_history.append(params[2])
            print("epoch: " + str(i+1) + "/" + str(self.config["simulation"]["num_epochs"]))
        
        self.plot_history([kp_history, ki_history, kd_history],'PID Parameters over Epochs','Epoch','Parameter Value',['kp', 'ki', 'kd'])
        self.plot_history(mse_history, 'Mean Squared Error over Epochs', 'Epoch', 'MSE Value', ['MSE'])
        print(error_history)
        self.plot_history(error_history, 'Error History over Timesteps', 'Timestep', 'Error', ['Error per timestep'])


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
            controller = NeuralNetworkController( config['simulation']['learning_rate'], config['controller']['neural_net']["activation_function"], params )
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
