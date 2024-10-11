import argparse

parser = argparse.ArgumentParser(description='PerfectGuan: PyTorch GuanDan AI')

# General Settings
parser.add_argument('--xpid', default='perfectguan',
                    help='Experiment id (default: perfectguan)')
parser.add_argument('--save_interval', default=180, type=int,
                    help='Time interval (in minutes) at which to save the model')    
parser.add_argument('--objective', default='wp', type=str, choices=['adp', 'wp', 'logadp'],
                    help='Use ADP or WP as reward (default: WP)')    

# Training settings
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='4,5,6,7', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=3, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=4, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--load_epoch', default=-1, type=int,
                    help='The epoch of model to load')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='perfectguan_checkpoints',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.5, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=8192, type=int,
                    help='Learner batch size')
parser.add_argument('--mini_batch_size', default=2048, type=int,
                    help='Learner mini batch size')
parser.add_argument('--unroll_length', default=128, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=128, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=0.5, type=float,
                    help='Max norm of gradients')
parser.add_argument('--GAE_lambda', default=0.95, type=float,
                    help='lambda of GAE')
parser.add_argument('--TD_gamma', default=0.99, type=float,
                    help='gamma of TD')
parser.add_argument('--intermediate_reward_scale', default=0.01, type=float,
                    help='intermediate reward scale')
parser.add_argument('--min_max_scale', default=0.25, type=float,
                    help='team min step max step scale')
parser.add_argument('--expert_cloning_scale', default=0, type=float,
                    help='behavior cloning scale')
parser.add_argument('--entropy_scale', default=0.1, type=float,
                    help='entropy of policy scale')
parser.add_argument('--epoch_per_batch', default=1, type=int,
                    help='train epoch for each batch')
parser.add_argument('--ppo_clip_value', default=0.2, type=float,
                    help='ppo clip value')
parser.add_argument('--normalize_advantage', default=True, type=bool,
                    help='whether use advantage normalization')

# Optimizer settings
parser.add_argument('--learning_rate', default=3e-4, type=float,
                    help='Learning rate')
