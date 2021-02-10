import numpy as np

# # Exercise 2 - Monte-carlo learning
# 
# We will use a simple random policy function to evaluate the state values. 
# 
# First of all, create a random policy function that just picks random actions (check Lab 01).
# 
# Then, implement a MC learning class that allows to learn the values based on full 
# rollouts of the policy in the environment.
# 
# Finally, you can generate rollouts of your policy in an environment, and update the 
# values using MC-learning.

class MC_Learning:
   
    def __init__(self, envir, gamma):
        self.size_environment = envir.size
        self.gamma = gamma
        self.coord_to_index_state = envir.coord_to_index_state
        self.values = np.zeros( (self.size_environment*self.size_environment) )
        self.counter = np.zeros( (self.size_environment*self.size_environment) )
    
    def update_values(self, rollout):
        # Calculate returns by going backwards
        g = 0
    
        rollout_with_returns = []
        for s, r in rollout[::-1]:
            
            g = g*self.gamma + r
            
            rollout_with_returns.append( (s,g))
        
        rollout_with_returns = rollout_with_returns[::-1] 
            
        # Update the values
        for s, g in rollout_with_returns:
            self.counter[s] += 1
            self.values[s] += 1/self.counter[s]*( g - self.values[s] )
        
    def display_values(self):
        value_matrix = np.zeros( (self.size_environment, self.size_environment) )
        for i in range(self.size_environment):
            for j in range(self.size_environment):
                state = self.coord_to_index_state[i, j]
                value_matrix[i,j] = self.values[state]
        return value_matrix