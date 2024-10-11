import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

# from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_buffers, create_optimizers, act

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['p1', 'p2', 'p3', 'p4']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss


def compute_GAE(model, position, state, legal_action_id, num_legal_actions, action_id, reward, next_state, done, flags):
    GAE_lambda = flags.GAE_lambda
    TD_gamma = flags.TD_gamma

    value = model.forward(position, state, num_legal_actions, legal_action_id, 'value')
    next_value = model.forward(position, next_state, num_legal_actions, legal_action_id, 'value')
    coeff_adv = TD_gamma * (1.0 - done.unsqueeze(1))
    value_target = reward.unsqueeze(1) + next_value * coeff_adv
    delta = value_target - value
    delta = delta.cpu().detach().numpy()
    coeff_adv = coeff_adv.cpu().detach().numpy()

    advantage_list = np.zeros_like(delta)
    advantage = 0.0
    for step in reversed(range(len(delta))):
        advantage = GAE_lambda * coeff_adv[step] * advantage + delta[step]
        advantage_list[step] = advantage
    advantage = torch.as_tensor(advantage_list, device=action_id.device)
    returns = value + advantage
    returns = returns.detach()
    return advantage, returns


def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device('cuda:'+str(flags.training_device))
    else:
        device = torch.device('cpu')

    done = batch['done'].to(device)
    episode_returns = batch['episode_return'].to(device)
    state = batch['state'].to(device)
    legal_action_id = batch['legal_action_id'].to(device)
    num_legal_actions = batch['num_legal_actions'].to(device)
    policy = batch['policy'].to(device)
    action_id = batch['action_id'].to(device)
    expert_action_id = batch['expert_action_id'].to(device)
    next_state = batch['next_state'].to(device)
    reward = batch['reward'].to(device)

    reward_mixed = episode_returns + reward * flags.intermediate_reward_scale
    action_id = action_id.unsqueeze(1)
    expert_action_id = expert_action_id.unsqueeze(1)
    episode_returns = episode_returns[done.to(torch.bool).squeeze()]
    mean_episode_return_buf[position].append(torch.mean(episode_returns))
        
    with lock:
        advantage, returns = compute_GAE(model, position, state, num_legal_actions, legal_action_id, 
                                         action_id, reward_mixed, next_state, done, flags)
        if flags.normalize_advantage and len(advantage) > 1:
            advantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + 1e-8)
        old_log_policy = policy
        returns = returns.detach()

        policy_loss_minibatch_list = []
        value_loss_minibatch_list = []
        policy_entropy_list = []

        for _ in range(flags.epoch_per_batch):
            indexs = np.random.permutation(flags.batch_size)
            indexs = torch.as_tensor(indexs)
            chunk_num = int(flags.batch_size / flags.mini_batch_size)
            minibatchs = torch.chunk(indexs, chunk_num, dim=0)
            for minibatch in minibatchs:
                state_minibatch = state[minibatch]
                legal_action_id_minibatch = legal_action_id[minibatch]
                num_legal_actions_minibatch = num_legal_actions[minibatch]
                action_id_minibatch = action_id[minibatch]
                expert_action_id_minibatch = expert_action_id[minibatch]
                old_log_policy_minibatch = old_log_policy[minibatch]
                advantage_minibatch = advantage[minibatch]
                returns_minibatch = returns[minibatch]

                log_policy_minibatch = model.forward(position, state_minibatch, num_legal_actions_minibatch, legal_action_id_minibatch, 'policy')
                policy_agent_minibatch = torch.gather(log_policy_minibatch, 1, action_id_minibatch)
                policy_expert_minibatch = torch.gather(log_policy_minibatch, 1, expert_action_id_minibatch)
                policy_expert_minibatch = policy_expert_minibatch + 1e-4 * torch.ones_like(policy_expert_minibatch)
                ratio_minibatch = torch.exp(policy_agent_minibatch - old_log_policy_minibatch)
                surr1_minibatch = ratio_minibatch * advantage_minibatch
                surr2_minibatch = torch.clamp(ratio_minibatch, 1-flags.ppo_clip_value, 1+flags.ppo_clip_value) * advantage_minibatch
                entropy_loss_minibatch = -torch.mean(-log_policy_minibatch)
                with torch.no_grad():
                    bc_loss_minibatch = torch.mean(-torch.log(policy_expert_minibatch))

                policy_loss_minibatch = -torch.mean(torch.min(surr1_minibatch, surr2_minibatch))
                policy_loss_minibatch = policy_loss_minibatch + bc_loss_minibatch * flags.expert_cloning_scale + entropy_loss_minibatch * flags.entropy_scale
                value_now_minibatch = model.forward(position, state_minibatch, num_legal_actions_minibatch, legal_action_id_minibatch, 'value')
                value_loss_minibatch = torch.mean(F.mse_loss(value_now_minibatch, returns_minibatch))

                policy_loss_minibatch_list.append(policy_loss_minibatch.item())
                value_loss_minibatch_list.append(value_loss_minibatch.item())
                policy_entropy_list.append(entropy_loss_minibatch.item())

                optimizer[position]['value'].zero_grad()
                optimizer[position]['policy'].zero_grad()

                value_loss_minibatch.backward()
                policy_loss_minibatch.backward()

                nn.utils.clip_grad_norm_(model.parameters(position, 'value'), flags.max_grad_norm)
                nn.utils.clip_grad_norm_(model.parameters(position, 'policy'), flags.max_grad_norm)

                optimizer[position]['value'].step()
                optimizer[position]['policy'].step()

        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_policy_'+position: np.mean(policy_loss_minibatch_list),
            'loss_value_'+position: np.mean(value_loss_minibatch_list),
            'entropy_policy_'+position: np.mean(policy_entropy_list)
        }

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.get_model(position).state_dict())
        return stats

def train(flags):  
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError("CUDA not available. \
                                 If you have GPUs, please specify the ID after `--gpu_devices`. \
                                 Otherwise, please train with CPU with \
                                 `python3 train.py --actor_device_cpu --training_device cpu`")

    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(1,flags.num_actor_devices+1)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers
    buffers = create_buffers(flags, device_iterator)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}
        
    for device in device_iterator:
        _free_queue = {'p1': ctx.SimpleQueue(), 'p2': ctx.SimpleQueue(), 'p3': ctx.SimpleQueue(), 'p4': ctx.SimpleQueue()}
        _full_queue = {'p1': ctx.SimpleQueue(), 'p2': ctx.SimpleQueue(), 'p3': ctx.SimpleQueue(), 'p4': ctx.SimpleQueue()}
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_p1',
        'loss_policy_p1',
        'loss_value_p1',
        'entropy_policy_p1',
        'mean_episode_return_p2',
        'loss_policy_p2',
        'loss_value_p2',
        'entropy_policy_p2',
        'mean_episode_return_p3',
        'loss_policy_p3',
        'loss_value_p3',
        'entropy_policy_p3',
        'mean_episode_return_p4',
        'loss_policy_p4',
        'loss_value_p4',
        'entropy_policy_p4',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'p1':0, 'p2':0, 'p3':0, 'p4':0}

    # Load models if any
    if flags.load_model:
        if flags.load_epoch == -1 and os.path.exists(checkpointpath):
            checkpoint_states = torch.load(
                checkpointpath, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu")
            )
            for k in ['p1', 'p2', 'p3', 'p4']:
                learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
                optimizers[k]['policy'].load_state_dict(checkpoint_states["policy_optimizer_state_dict"][k])
                optimizers[k]['value'].load_state_dict(checkpoint_states["value_optimizer_state_dict"][k])
                for device in device_iterator:
                    models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
            stats = checkpoint_states["stats"]
            frames = checkpoint_states["frames"]
            position_frames = checkpoint_states["position_frames"]
            log.info(f"Resuming preempted job, current stats:\n{stats}")
        else:
            frames = flags.load_epoch
            for k in ['p1', 'p2', 'p3', 'p4']:
                checkpointpath_tmp = 'perfectguan_checkpoints/perfectguan/{}_weights_{}.ckpt'.format(k, flags.load_epoch)
                checkpoint_states = torch.load(
                    checkpointpath_tmp, map_location=("cuda:"+str(flags.training_device) if flags.training_device != "cpu" else "cpu"))
                learner_model.get_model(k).load_state_dict(checkpoint_states)
                for device in device_iterator:
                    models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())

    # Starting actor processes
    for device in device_iterator:
        for i in range(flags.num_actors):
            # act(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags)
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model, batch, 
                optimizers, flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                frames += T
                position_frames[position] += T

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['p1'].put(m)
            free_queue[device]['p2'].put(m)
            free_queue[device]['p3'].put(m)
            free_queue[device]['p4'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'p1': threading.Lock(), 'p2': threading.Lock(), 'p3': threading.Lock(), 'p4': threading.Lock()}
    position_locks = {'p1': threading.Lock(), 'p2': threading.Lock(), 'p3': threading.Lock(), 'p4': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['p1', 'p2', 'p3', 'p4']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, 
                    args=(i,device,position,locks[device][position],position_locks[position]))
                thread.start()
                threads.append(thread)
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        if not os.path.isdir(flags.savedir):
            os.mkdir(flags.savedir)
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'policy_optimizer_state_dict': {k: optimizers[k]['policy'].state_dict() for k in optimizers},
            'value_optimizer_state_dict': {k: optimizers[k]['value'].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['p1', 'p2', 'p3', 'p4']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(60)

            if timer() - last_checkpoint_time > flags.save_interval * 60:  
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 5000:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.info('After %i (p1:%i p2:%i p3:%i p4:%i) frames: @ %.1f fps (avg@ %.1f fps) (p1:%.1f p2:%.1f p3:%.1f p4:%.1f) Stats:\n%s',
                     frames,
                     position_frames['p1'],
                     position_frames['p2'],
                     position_frames['p3'],
                     position_frames['p4'],
                     fps,
                     fps_avg,
                     position_fps['p1'],
                     position_fps['p2'],
                     position_fps['p3'],
                     position_fps['p4'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    # plogger.close()
