import os
import warnings
warnings.filterwarnings("ignore", category=Warning)

from perfectguan.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train(flags)
