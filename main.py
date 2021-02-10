from dungeon.dungeon import Dungeon
import numpy as np
import random
from action import Action

def main():
    action = random_policy()
    
    
# We will use a simple random policy function to evaluate the state values. 
# 
# First of all, create a random policy function that just picks random actions (check Lab 01).
# 

def random_policy():
    rnd_no = random.randint(0, 3)
    return Action.actions_dict.get(rnd_no)
    

if __name__ == "__main__":
    main()    

