"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from perfectguan.config import MAX_ACTION, ACTION_SIZE
from .utils import DECK4P, POLICYNETINPUTSIZE, VALUENETINPUTSIZE

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(DECK4P, 128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(POLICYNETINPUTSIZE + 128, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, state, num_legal_actions, legal_action_id):
        state = state.to(torch.float32)
        x_imperfect = state[:,:1093].unsqueeze(1)
        x_imperfect = x_imperfect.repeat(1,MAX_ACTION,1)
        z = state[:,1498:3658].reshape(-1,5,432)
        legal_actions = state[:,3658:].reshape(-1,MAX_ACTION,ACTION_SIZE)
        x = torch.concat([x_imperfect,legal_actions],dim=-1)

        lstm_out, (_, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:].unsqueeze(1)
        lstm_out = lstm_out.repeat(1,MAX_ACTION,1)
        x = torch.cat([lstm_out,x], dim=-1)
        logit = self.mlp(x).squeeze(-1)

        mask = torch.zeros_like(logit).to(logit.device)
        for i in range(logit.shape[0]):
            mask[i,:num_legal_actions[i]] = 1
        logit = logit.unsqueeze(-1)

        onehot = F.one_hot(legal_action_id, 367)
        mask = (mask.unsqueeze(-1) * onehot).sum(1)
        mask = (mask > 0).to(torch.float32)
        mask = (1 - mask) * (-1e10)
        output = (logit * onehot).sum(1) + mask
        output = F.softmax(output, dim=1)
        return output
    
class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(DECK4P, 128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(VALUENETINPUTSIZE + 128, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, state):
        state = state.to(torch.float32)
        x_perfect = state[:,:1498]
        z = state[:,1498:3658].reshape(-1,5,432)
        x = x_perfect

        lstm_out, (_, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        value = self.mlp(x)
        return value

class PerfectGuanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = PolicyNet()
        self.value_net = ValueNet()

    def forward(self, state, num_legal_actions, legal_action_id, net_ref):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            num_legal_actions = num_legal_actions.unsqueeze(0)
            legal_action_id = legal_action_id.unsqueeze(0)
        if net_ref == 'policy':
            output = self.policy_net(state, num_legal_actions, legal_action_id)
        elif net_ref == 'value':
            output = self.value_net(state)
        return output

# Model dict is only used in evaluation but not training
model_dict = {}
model_dict['p1'] = PerfectGuanModel()
model_dict['p2'] = PerfectGuanModel()
model_dict['p3'] = PerfectGuanModel()
model_dict['p4'] = PerfectGuanModel()

class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.models['p1'] = PerfectGuanModel().to(torch.device(device))
        self.models['p2'] = PerfectGuanModel().to(torch.device(device))
        self.models['p3'] = PerfectGuanModel().to(torch.device(device))
        self.models['p4'] = PerfectGuanModel().to(torch.device(device))

    def forward(self, position, state, num_legal_actions, legal_action_id, net_ref):
        model = self.models[position]
        output = model.forward(state, num_legal_actions, legal_action_id, net_ref)
        return output

    def share_memory(self):
        self.models['p1'].share_memory()
        self.models['p2'].share_memory()
        self.models['p3'].share_memory()
        self.models['p4'].share_memory()

    def eval(self):
        self.models['p1'].eval()
        self.models['p2'].eval()
        self.models['p3'].eval()
        self.models['p4'].eval()

    def parameters(self, position, net_ref):
        if net_ref == 'policy':
            return self.models[position].policy_net.parameters()
        elif net_ref == 'value':
            return self.models[position].value_net.parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
    
# model = Model('cpu')
# breakpoint()
# inputs = torch.ones(512, 32738)
# model.forward('landlord', inputs, 'value')