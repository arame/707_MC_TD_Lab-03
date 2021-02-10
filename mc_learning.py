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
   
    def __init__(self, N):
        self.N = N
        self.values = np.zeros(N, N)
    
    def update_values(self, rollout):
        ...
        
    def display_values(self):
        ...