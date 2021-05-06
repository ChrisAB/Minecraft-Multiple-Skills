import copy


class ActionConverter:
    def __init__(self, actions):
        self.actions = actions
        self.actions_array = self.convert_to_array(actions)

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
                    my_dict[key][0] = 90
                elif(actions_array[i+1] == 1):
                    my_dict[key][0] = -90
                else:
                    my_dict[key][0] = 0
                i += 2
                if(actions_array[i] == 1):
                    my_dict[key][1] = 90
                elif(actions_array[i+1] == 1):
                    my_dict[key][1] = -90
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
