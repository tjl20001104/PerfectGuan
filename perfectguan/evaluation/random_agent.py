import random

class RandomAgent():

    def __init__(self):
        self.name = 'Random'

    def act(self, infoset):
        if len(infoset.legal_actions) == 0:
            return 0
        
        return random.choice(infoset.legal_actions)
