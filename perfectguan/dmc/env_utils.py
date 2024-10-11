"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch 
from perfectguan.env.env import _cards2array

def _format_observation(obs, u_dict, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)

    state = torch.from_numpy(obs['state'])
    legal_action_id = torch.from_numpy(obs['action_id'].astype(np.int64))
    num_legal_actions = obs['num_legal_actions']
    min_step_playout = torch.from_numpy(obs['min_step_all'])
    util_dict = torch.from_numpy(np.array(list(u_dict.values())).astype('float')).to(device)

    obs = {'state': state.to(device),
           'rank': obs['rank'],
           'legal_action_id': legal_action_id.to(device),
           'legal_actions': obs['legal_actions'],
           'num_legal_actions': torch.tensor(num_legal_actions).to(device)
           }

    return position, obs, state, legal_action_id, num_legal_actions, min_step_playout, util_dict

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        initial_position, initial_obs, state, legal_action_id, num_legal_actions, min_step_playout, _\
            = _format_observation(self.env.reset(), self.env.player_utility_dict, self.device)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_state=state,
            obs_legal_action_id=legal_action_id,
            obs_num_legal_actions=num_legal_actions,
            obs_min_step_playout=min_step_playout,
            utility_dict=torch.zeros((4, 1), dtype=torch.float64)
            )
        
    def step(self, action):
        obs, reward, done, u_dict = self.env.step(action)
        self.episode_return += reward
        episode_return = self.episode_return 

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs, state, legal_action_id, num_legal_actions, min_step_playout, _\
            = _format_observation(obs, u_dict, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        
        return position, obs, dict(
            done=done,
            episode_return=episode_return,
            obs_state=state,
            obs_legal_action_id=legal_action_id,
            obs_num_legal_actions=num_legal_actions,
            obs_min_step_playout=min_step_playout,
            utility_dict=torch.zeros((4, 1), dtype=torch.float64)
            )

    def close(self):
        self.env.close()
