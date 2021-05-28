from .action_converter import ActionConverter
import copy


class CraftWoodenPickaxeActionConverter(ActionConverter):
    def __init__(self, actions):
        super().__init__(actions)
        self.place_actions = ['crafting_table']
        self.craft_actions = ['stick', 'planks', 'crafting_table']
        self.craft_nearby_actions = ['wooden_pickaxe']

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
                    my_dict[key][1] = 25
                elif(actions_array[i+1] == 1):
                    my_dict[key][1] = -25
                else:
                    my_dict[key][1] = 0
                i += 2
            elif key == 'place':
                my_dict[key] = 'none'
                for action in self.place_actions:
                    if(actions_array[i] == 1):
                        my_dict[key] = action
                        i += 1
                        break
                    else:
                        i += 1
            elif key == 'craft':
                my_dict[key] = 'none'
                for action in self.craft_actions:
                    if(actions_array[i] == 1):
                        my_dict[key] = action
                        i += 1
                        break
                    else:
                        i += 1
            elif key == 'nearbycraft':
                my_dict[key] = 'none'
                for action in self.nearby_craft_actions:
                    if(actions_array[i] == 1):
                        my_dict[key] = action
                        i += 1
                        break
                    else:
                        i += 1
        return my_dict
