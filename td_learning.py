import numpy as np

# # Exercise 3: TD-learning
# 
# We will now compare the values obtained by MC learning with values obtained with TD learning.
# 
# Create a class that allows to update values every time a new state transition occurs.
# 

class TD_Learning():
    
    def __init__(self, envir, alpha, gamma):
        self.size_environment = envir.size
        self.alpha = alpha
        self.gamma = gamma
        self.coord_to_index_state = envir.coord_to_index_state
        self.values = np.zeros( (self.size_environment*self.size_environment) )
    
    def update_values(self, s_current, reward_next, s_next):
        self.values[s_current] = self.values[s_current] + self.alpha * ( reward_next + self.gamma*self.values[s_next] - self.values[s_current] )
        
    def display_values(self):
        value_matrix = np.zeros( (self.size_environment, self.size_environment) )
        for i in range(self.size_environment):
            for j in range(self.size_environment):

                state = self.coord_to_index_state[i, j]
                
                value_matrix[i,j] = self.values[state]
                
        return value_matrix
