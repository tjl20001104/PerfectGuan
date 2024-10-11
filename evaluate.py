import os 
import argparse

from perfectguan.evaluation.simulation import evaluate, eval_h

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Guan Dan Evaluation')

    parser.add_argument('--p1', type=str,
            default='guanzero')
    parser.add_argument('--p2', type=str,
            default='guanzero')
    parser.add_argument('--p3', type=str,
            default='guanzero')
    parser.add_argument('--p4', type=str,
            default='guanzero')
    parser.add_argument('--epoch', type=int,
            default=-1)
    
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='7')
    parser.add_argument('--role', type=str, default='none')  # play as p1, p2, p3, p4
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    if args.debug == 1:
        eval_h(args.p1,
               args.p2,
               args.p3,
               args.p4,
               args.epoch,
               args.eval_data,
               1,  # args.num_workers is ALWAYS 1
               args.role)
    
    if args.role in ['p1', 'p2', 'p3', 'p4', 'obs4']:
        eval_h(args.p1,
               args.p2,
               args.p3,
               args.p4,
               args.epoch,
               args.eval_data,  # defaults to 'eval_data.pkl'
               1,  # args.num_workers is ALWAYS 1
               args.role)
    else:
        evaluate(args.p1,
                 args.p2,
                 args.p3,
                 args.p4,
                 args.epoch,
                 args.eval_data,
                 args.num_workers)
        