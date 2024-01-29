import jax.numpy as jnp


class Bathtub:
    def __init__(self, config):
        self.gravitational_constant = config["plant"]["bathtub"]["gravitational_constant"]
        self.water_height = config["plant"]["bathtub"]["initial_height"]
        self.water_volume = config["plant"]["bathtub"]["initial_height"]*config["plant"]["bathtub"]["cross_sectional_area_A"]
        self.A = config["plant"]["bathtub"]["cross_sectional_area_A"]
        self.C = self.A/100

    def update_state(self, u, d):
        # print("HEIGHT BEFORE: ")
        # print(self.water_height)
        change_in_volume = u + d - jnp.sqrt(2*self.gravitational_constant*self.water_height)*self.C
        self.water_volume+= change_in_volume
        self.water_height+= change_in_volume/self.A
        # print("HEIGHT AFTER: ")
        # print(self.water_height)
        return self.water_height

    
class CournotCompetition:
    def __init__(self, config):
        self.q1 = config["plant"]["cournot_competition"]["q1"]
        self.q2 = config["plant"]["cournot_competition"]["q2"]
        self.p_max = config["plant"]["cournot_competition"]["maximum_price_pmax"]
        self.cm = config["plant"]["cournot_competition"]["marginal_cost_cm"]

    def update_state(self, u, d):
        self.q1+=u
        self.q2+=d
        if self.q1 >1:
            self.q1 = 1
        elif self.q1 <0:
            self.q1 = 0
        if self.q2 >1:
            self.q2 = 1
        elif self.q2 <0:
            self.q2 = 0
        
        q = self.q1 + self.q2
        p = self.p_max - q
        P1 = self.q1*(p-self.cm)
        return P1
    
class LogisticGrowthPlant:
    def __init__(self, config):
        self.population = config["plant"]["logistic_growth"]["initial_population"]
        self.growth_rate = config["plant"]["logistic_growth"]["growth_rate"]
        self.carrying_capacity = config["plant"]["logistic_growth"]["carrying_capacity"]

    def update_state(self, control_signal, noise):
        growth = self.growth_rate * self.population * (1 - self.population / self.carrying_capacity)
        new_population = self.population + growth + control_signal + noise
        new_population = max(min(new_population, self.carrying_capacity), 0)
        self.population = new_population
        return self.population
    
    

