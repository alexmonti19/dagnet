import argparse
import pathlib
import torch
import random
import logging
import numpy as np

from models.baseline.model import VRNN
from models.baseline.data.loader_bsk import data_loader
from models.utils.utils import average_displacement_error, final_displacement_error, relative_to_abs

parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--num_workers', default=0, type=int, required=False, help='Number of workers for loading data')
parser.add_argument('--obs_len', default=10, type=int, required=False, help='Timesteps of observation')
parser.add_argument('--pred_len', default=40, type=int, required=False, help='Timesteps of prediction')
parser.add_argument('--players', type=str, choices=['atk', 'def', 'all'], required=False, help='Which players to use')

# Model options
parser.add_argument('--clip', default=10, type=int, required=False, help='Gradient clipping')
parser.add_argument('--n_layers', default=2, type=int, required=False, help='Number of recurrent layers')
parser.add_argument('--x_dim', default=2, type=int, required=False, help='Dimension of the input of the single agent')
parser.add_argument('--h_dim', default=64, type=int, required=False, help='Dimension of the hidden layers')
parser.add_argument('--z_dim', default=96, type=int, required=False, help='Dimension of the latent variables')
parser.add_argument('--rnn_dim', default=200, type=int, required=False, help='Dimension of the recurrent layer')

# Miscellaneous
parser.add_argument('--batch_size', default=32, type=int, required=False, help='Batch size')
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples for evaluation')
parser.add_argument('--seed', default=128, type=int, required=False, help='PyTorch random seed')
parser.add_argument('--run', required=True, type=str, help="Which run evaluate")
parser.add_argument('--best', default=False, action='store_true', help='Evaluate with best checkpoint')
parser.add_argument('--epoch', type=int, help='Evaluate with the checkpoint of a specific epoch')


# dirs
ROOT_DIR = pathlib.Path('.').absolute().parent.parent
BASE_DIR = ROOT_DIR / 'runs' / 'baseline'
DATASET_DIR = BASE_DIR.parent.parent.absolute() / 'datasets' / 'basket'

# global vars
FORMAT = '[%(levelname)s | %(asctime)s]: %(message)s'
DATEFMT = '%Y/%m/%d %H:%M'


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, model, loader, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = [tensor.cuda() for tensor in data]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt,
             _, _, seq_start_end) = data

            _, _, h = model(obs_traj_rel)

            ade, fde = [], []
            total_traj += obs_traj.shape[1]

            for _ in range(num_samples):
                samples_rel = model.sample(args.pred_len, h)
                samples = relative_to_abs(samples_rel, obs_traj[-1])

                ade.append(average_displacement_error(samples, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(samples[-1, :, :], pred_traj_gt[-1, :, :], mode='raw'))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / total_traj

    return ade, fde


def main(args):
    curr_run_dir = BASE_DIR / '{}'.format(args.run)
    saves_dir = curr_run_dir / 'saves'
    saves_best = saves_dir / 'best'

    # logging
    logfile_path = curr_run_dir.absolute() / '{}_eval.log'.format(args.run)
    logging.basicConfig(filename=logfile_path, level=logging.INFO, filemode='w', format=FORMAT, datefmt=DATEFMT)

    if args.best:
        # load best checkpoint
        save = list(pathlib.Path(saves_best).glob('*.pth'))[-1]
    elif args.epoch is not None:
        # load checkpoint from specific epoch
        save = saves_dir.absolute() / 'checkpoint_epoch_{}.pth'.format(args.epoch)
        if not pathlib.Path(save).is_file():
            raise(Exception("Couldn't find a checkpoint for the specified epoch"))
    else:
        # load last checkpoint
        save = list(pathlib.Path(saves_dir).glob('*.pth'))[-1]

    logging.info('Loading checkpoint...')
    checkpoint = torch.load(save)
    args.__dict__ = checkpoint['args']

    # data
    logging.info('Loading training/validation/test sets...')
    _, train_loader = data_loader(args, DATASET_DIR, 'train')
    _, valid_loader = data_loader(args, DATASET_DIR, 'validation')
    _, test_loader = data_loader(args, DATASET_DIR, 'test')

    model = VRNN(args).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info('Evaluating...')

    ade, fde = evaluate(args, model, test_loader, args.num_samples)

    logging.info("### [VRNN] BASKET-SPORTVU RESULTS ###")
    logging.info("Pred Len: {}, ADE: {:.6f}, FDE: {:.6f}".format(args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
