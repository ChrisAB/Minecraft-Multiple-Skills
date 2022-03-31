import copy
import math
import torch
import numpy as np


class ActionConverter:
    def __init__(self, actions):
        self.actions = actions
        self.actions_array = self.convert_to_array(actions)
        self.n_actions = len(self.actions_array)
        self.current_action_array = []
        self.current_action_tensor = None

    def convert_to_array(self, actions):
        actions_array = []
        for key, value in actions.items():
            if key == 'attack' or key == 'jump' or key == 'sneak' or key == 'sprint' or key == 'back' or key == 'forward' or key == 'left' or key == 'right':
                actions_array += [value]
            elif key == 'camera':
                if(value[0] > 0):
                    actions_array += [1, 0]
                elif(value[0] < 0):
                    actions_array += [0, 1]
                else:
                    actions_array += [0, 0]
                if(value[1] > 0):
                    actions_array += [1, 0]
                elif(value[1] < 0):
                    actions_array += [0, 1]
                else:
                    actions_array += [0, 0]
            elif key == 'place':
                if (value == 'none'):
                    actions_array += [0]
                else:
                    actions_array += [1]
        return actions_array

    def convert_to_ordereddict(self, actions_array):
        i = 0
        my_dict = copy.deepcopy(self.actions)
        for key, value in my_dict.items():
            if key == 'attack' or key == 'jump' or key == 'sneak' or key == 'sprint' or key == 'back' or key == 'forward' or key == 'left' or key == 'right':
                my_dict[key] = actions_array[i]
                i += 1
            elif key == 'camera':
                if(actions_array[i] == 1):
                    self.actions[key][0] = min(self.actions[key][0]+90, 180)
                    my_dict[key][0] = self.actions[key][0]
                elif(actions_array[i+1] == 1):
                    self.actions[key][0] = max(self.actions[key][0]-90, -180)
                    my_dict[key][0] = self.actions[key][0]
                else:
                    my_dict[key][0] = self.actions[key][0]
                i += 2
                if(actions_array[i] == 1):
                    my_dict[key][1] = 25
                elif(actions_array[i+1] == 1):
                    my_dict[key][1] = -25
                else:
                    my_dict[key][1] = 0
                i += 2
            elif key == 'place':
                if (actions_array[i] == 0):
                    my_dict[key] = 'none'
                else:
                    my_dict[key] = 'dirt'
                i += 1
        return my_dict

    def get_action(self, action_tensor):
        self.current_action_tensor = action_tensor
        max_index = self.current_action_tensor.argmax()
        self.current_action_array = np.zeros(self.n_actions)
        self.current_action_array[max_index] = 1
        return self.convert_to_ordereddict(self.current_action_array)
