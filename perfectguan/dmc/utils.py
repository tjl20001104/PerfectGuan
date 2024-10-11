import typing
import logging
import traceback
import numpy as np
from collections import  defaultdict

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from perfectguan.env import Env
from perfectguan.env.stepwise_reward import decode_action
from perfectguan.config import MAX_ACTION, ACTION_SIZE
from ..evaluation.rule_agent import RuleAgent_2

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

DECK4P = 432  # was: 162 (3*54)
IMPERFECTSTATESIZE = 1093 # 1102 - 9
POLICYNETINPUTSIZE = IMPERFECTSTATESIZE + ACTION_SIZE
PERFECTSTATESIZE = 1498 # 1507 - 9
VALUENETINPUTSIZE = PERFECTSTATESIZE
STATESIZE = PERFECTSTATESIZE + 5 * DECK4P + MAX_ACTION * ACTION_SIZE

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    acq_buffer_num = int(flags.batch_size / flags.unroll_length)
    with lock:
        indices = [full_queue.get() for _ in range(acq_buffer_num)]
    batch = {
        key: torch.cat([buffers[key][m] for m in indices], dim=0)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = ['p1', 'p2', 'p3', 'p4']
    optimizers = defaultdict(dict)
    for position in positions:
        optimizer_policy = torch.optim.AdamW(
            learner_model.parameters(position, 'policy'),
            lr=flags.learning_rate,
            eps=1e-7)
        optimizer_value = torch.optim.AdamW(
            learner_model.parameters(position, 'value'),
            lr=flags.learning_rate,
            eps=1e-7)
        optimizers[position]['policy'] = optimizer_policy
        optimizers[position]['value'] = optimizer_value

    return optimizers

def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = ['p1', 'p2', 'p3', 'p4']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            specs = dict(
                done=dict(size=(T,), dtype=torch.int8),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                state=dict(size=(T, STATESIZE), dtype=torch.bool),
                legal_action_id=dict(size=(T, MAX_ACTION), dtype=torch.int64),
                num_legal_actions=dict(size=(T,), dtype=torch.int64),
                policy=dict(size=(T,), dtype=torch.float32),
                action_id=dict(size=(T,), dtype=torch.int64),
                expert_action_id=dict(size=(T,), dtype=torch.int64),
                next_state=dict(size=(T, STATESIZE), dtype=torch.bool),
                reward=dict(size=(T,), dtype=torch.float32),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = ['p1', 'p2', 'p3', 'p4']
    if flags.expert_cloning_scale > 0:
        expert_agent = RuleAgent_2()
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        state_buf = {p: [] for p in positions}
        legal_action_id_buf = {p: [] for p in positions}
        num_legal_actions_buf = {p: [] for p in positions}
        policy_buf = {p: [] for p in positions}
        action_id_buf = {p: [] for p in positions}
        expert_action_id_buf = {p: [] for p in positions}
        next_state_buf = {p: [] for p in positions}
        reward_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()
        old_min_step_all = env_output['obs_min_step_playout']

        while True:
            while True:
                state_buf[position].append(env_output['obs_state'])
                legal_action_id_buf[position].append(env_output['obs_legal_action_id'])
                num_legal_actions_buf[position].append(env_output['obs_num_legal_actions'])

                with torch.no_grad():
                    logit = model.forward(position, obs['state'], obs['num_legal_actions'], obs['legal_action_id'], 'policy')
                logit = logit.squeeze(0)
                if flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                    legal_actions = env_output['obs_legal_action_id'][:env_output['obs_num_legal_actions']]
                    action_id = legal_actions[torch.randint(legal_actions.shape[0],(1,))]
                else:
                    action_id = torch.argmax(logit,dim=0)
                action_id = action_id.item()
                action = decode_action(action_id, obs['legal_actions'], obs['rank'])
                policy_buf[position].append(logit[action_id])
                if flags.expert_cloning_scale > 0:
                    expert_action_id = expert_agent.act(env.env.infoset, return_idx=True)
                    if expert_action_id == []:
                        expert_action_id_buf[position].append(366)
                    else:
                        expert_action_id_buf[position].append(expert_action_id)
                else:
                    expert_action_id_buf[position].append(0)

                action_id_buf[position].append(action_id)
                size[position] += 1
                
                position_next, obs, env_output = env.step(action)
                next_state_buf[position].append(env_output['obs_state'])
                position = position_next

                if position == 'p1':
                    new_min_step_all = env_output['obs_min_step_playout']
                    team1_old_min = min(old_min_step_all[0],old_min_step_all[2])
                    team2_old_min = min(old_min_step_all[1],old_min_step_all[3])
                    team1_new_min = min(new_min_step_all[0],new_min_step_all[2])
                    team2_new_min = min(new_min_step_all[1],new_min_step_all[3])
                    team1_old_max = max(old_min_step_all[0],old_min_step_all[2])
                    team2_old_max = max(old_min_step_all[1],old_min_step_all[3])
                    team1_new_max = max(new_min_step_all[0],new_min_step_all[2])
                    team2_new_max = max(new_min_step_all[1],new_min_step_all[3])
                    old_relative_dis = 1.0*(team1_old_min-team2_old_min)+flags.min_max_scale*(team1_old_max-team2_old_max)
                    new_relative_dis = 1.0*(team1_new_min-team2_new_min)+flags.min_max_scale*(team1_new_max-team2_new_max)
                    distance_change = new_relative_dis - old_relative_dis
                    reward_buf['p1'].append((-1.0) * distance_change)
                    reward_buf['p2'].append((+1.0) * distance_change)
                    reward_buf['p3'].append((-1.0) * distance_change)
                    reward_buf['p4'].append((+1.0) * distance_change)
                    old_min_step_all = new_min_step_all

                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(reward_buf[p])
                        if diff > 0:
                            reward_buf[p].extend([0.0] * diff)
                        elif diff < 0:
                            reward_buf[p].pop(-1)

                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] if p == 'p1' or p == 'p3' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break
            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['state'][index][t, ...] = state_buf[p][t]
                        buffers[p]['legal_action_id'][index][t, ...] = legal_action_id_buf[p][t]
                        buffers[p]['num_legal_actions'][index][t, ...] = num_legal_actions_buf[p][t]
                        buffers[p]['policy'][index][t, ...] = policy_buf[p][t]
                        buffers[p]['action_id'][index][t, ...] = action_id_buf[p][t]
                        buffers[p]['expert_action_id'][index][t, ...] = expert_action_id_buf[p][t]
                        buffers[p]['next_state'][index][t, ...] = next_state_buf[p][t]
                        buffers[p]['reward'][index][t, ...] = reward_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    legal_action_id_buf[p] = legal_action_id_buf[p][T:]
                    num_legal_actions_buf[p] = num_legal_actions_buf[p][T:]
                    policy_buf[p] = policy_buf[p][T:]
                    action_id_buf[p] = action_id_buf[p][T:]
                    expert_action_id_buf[p] = expert_action_id_buf[p][T:]
                    next_state_buf[p] = next_state_buf[p][T:]
                    reward_buf[p] = reward_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
