from dungeon.dungeon import Dungeon
import numpy as np
import matplotlib.pyplot as plt
import random
from action import Action
from ice_dungeon import IceDungeon
from mc_learning import MC_Learning
from td_learning import TD_Learning

def main_mc():
    size = 20
    dungeon = IceDungeon(size)
    gamma = 0.99
    mc = MC_Learning(dungeon, gamma)

    dungeon.reset()
    dungeon.display()
    # Run this cell multiple time

    rol = perform_rollout(dungeon)
    mc.update_values(rol)

    vals = mc.display_values()
    plt.imshow( (vals - vals.min())/(vals.max() - vals.min()) )
    plt.show()

    mc = MC_Learning(dungeon, 0.99)
    for _ in range(1000):
        rol = perform_rollout(dungeon)
        mc.update_values(rol)

    vals = mc.display_values()
    plt.imshow( (vals - vals.min())/(vals.max() - vals.min()) )
    plt.show()
    
# We will use a simple random policy function to evaluate the state values. 
# 
# First of all, create a random policy function that just picks random actions (check Lab 01).
# 

def random_policy(a):
    rnd_no = random.randint(0, 3)
    return a.index_to_actions.get(rnd_no)
    
def perform_rollout(envir):
    a = Action()
    s = envir.reset()
    done = False
    
    rollout = []
    while not done:
        action = random_policy(a)
        s_next, r, done = envir.step(action)
        rollout.append((s, r))
        s = s_next
    
    return rollout

#############################################################################################
def main_td():
    size = 20
    dungeon = IceDungeon(size)
    td = TD_Learning(dungeon, 0.1, 0.99)
    dungeon.reset()
    dungeon.display()
    td_learning_episode(td, dungeon)
    vals = td.display_values()
    plt.imshow( (vals - vals.min())/(vals.max() - vals.min()) )
    plt.show()
    td = TD_Learning(dungeon, 0.01, 0.99)
    for _ in range(1000):
        td_learning_episode(td, dungeon)

    vals = td.display_values()
    plt.imshow( (vals - vals.min())/(vals.max() - vals.min()) )
    plt.show()

def td_learning_episode(td, envir):
    s = envir.reset()
    done = False
    
    while not done:
        action = random_policy(s)
        s_next, r, done = envir.step(action)
        td.update_values(s, r, s_next)
        s = s_next

if __name__ == "__main__":
    main_mc() 
    # main_td()   

