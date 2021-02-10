from collections import namedtuple

class Action:
    def __init__(self):
        # COnvenient data structure to hold information about actions
        action_tuple = namedtuple('Action', 'name index delta_i delta_j')
            
        self.up = action_tuple('up', 0, -1, 0)    
        self.down = action_tuple('down', 1, 1, 0)    
        self.left = action_tuple('left', 2, 0, -1)    
        self.right = action_tuple('right', 3, 0, 1)    

        self.index_to_actions = {}
        for action in [self.up, self.down, self.left, self.right]:
            self.index_to_actions[action.index] = action