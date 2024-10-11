import os
import torch
import numpy as np

from perfectguan.env.stepwise_reward import decode_action
from perfectguan.env.env import get_obs
from perfectguan.dmc.models import PerfectGuanModel


def _load_model(position, epoch):
    if epoch == -1:
        model_dir = 'perfectguan_checkpoints/perfectguan'
        checkpointpath = '{}/model.tar'.format(model_dir)
        checkpoint_states = torch.load(checkpointpath,map_location=("cuda:0"))
        model = PerfectGuanModel()
        model.load_state_dict(checkpoint_states["model_state_dict"][position])
    else:
        checkpointpath = 'perfectguan_checkpoints/perfectguan/{}_weights_{}.ckpt'.format(position, epoch)
        checkpoint_states = torch.load(checkpointpath,map_location=("cuda:0"))
        model = PerfectGuanModel()
        model.load_state_dict(checkpoint_states)
    return model

class PerfectGuanAgent:
    def __init__(self, position, epoch):
        self.model = _load_model(position, epoch)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = get_obs(infoset) 

        state = torch.from_numpy(obs['state']).to(self.device)
        legal_action_id = torch.from_numpy(obs['action_id'].astype(np.int64)).to(self.device)
        num_legal_actions = torch.tensor(obs['num_legal_actions']).to(self.device)
        legal_actions = obs['legal_actions']
        rank = obs['rank']

        with torch.no_grad():
            logit = self.model.forward(state, num_legal_actions, legal_action_id, 'policy')
        logit = logit.squeeze(0)
        best_action_id = torch.argmax(logit)
        best_action = decode_action(best_action_id, legal_actions, rank)
        return best_action
