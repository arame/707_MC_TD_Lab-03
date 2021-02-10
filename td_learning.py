

# # Exercise 3: TD-learning
# 
# We will now compare the values obtained by MC learning with values obtained with TD learning.
# 
# Create a class that allows to update values every time a new state transition occurs.
# 

class TD_Learning():
    
    def __init__(self, N):
        
        self.values = ...
    
    def update_values(self, s_current, reward_next, s_next):
        ...
        
    def display_values(self):
        ...