

class ActionConverter:
    def __init__(self, actions):
        self.actions = actions
        self.actions_array = self.convert_to_array()

    def convert_to_array(self):
        for key, value in self.actions.items():
            print(key, value)

    def convert_to_ordereddict(self):
        i = 0
        for key, value in self.actions.items():
            print(key, value, actions_array[i])
            i += 1
