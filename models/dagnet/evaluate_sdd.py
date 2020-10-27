import torch
import argparse
import pathlib
import random
import logging
import numpy as np

from models.dagnet.model import DAGNet
from models.dagnet.data.loader_sdd import data_loader
from models.utils.adjacency_matrix import compute_adjs_distsim, compute_adjs_knnsim, compute_adjs
from models.utils.utils import relative_to_abs, average_displacement_error, final_displacement_error, to_goals_one_hot


parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--num_workers', default=0, type=int, required=False, help='Number of workers for loading data')
parser.add_argument('--obs_len', default=8, type=int, required=False, help='Timesteps of observation')
parser.add_argument('--pred_len', default=12, type=int, required=False, help='Timesteps of prediction')
parser.add_argument('--skip', default=1, type=int, required=False, help='Step for skipping frames')
parser.add_argument('--n_cells_x', default=32, type=int, required=False, help='Grid cells along x')
parser.add_argument('--n_cells_y', default=30, type=int, required=False, help='Grid cells along y')
parser.add_argument('--goals_window', default=4, type=int, required=False, help='How many timesteps compute goals')

# Model
parser.add_argument('--clip', default=10, type=int, required=False, help='Gradient clipping')
parser.add_argument('--n_layers', default=1, type=int, required=False, help='Number of recurrent layers')
parser.add_argument('--x_dim', default=2, type=int, required=False, help='Dimension of the input of the single agent')
parser.add_argument('--h_dim', default=64, type=int, required=False, help='Dimension of the hidden layers')
parser.add_argument('--z_dim', default=32, type=int, required=False, help='Dimension of the latent variables')
parser.add_argument('--rnn_dim', default=256, type=int, required=False, help='Dimension of the recurrent layers for single agents')

# Miscellaneous
parser.add_argument('--batch_size', default=16, type=int, required=False, help='Batch size')
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples for evaluation')
parser.add_argument('--seed', default=128, type=int, required=False, help='PyTorch random seed')
parser.add_argument('--run', required=True, type=str, help='Which run evaluate')
parser.add_argument('--best', default=False, action='store_true', help='Evaluate with best checkpoint')
parser.add_argument('--epoch', type=int, help='Evaluate with the checkpoint of a specific epoch')

# Graph
parser.add_argument('--graph_model', type=str, required=False, choices=['gat','gcn'], help='Graph type')
parser.add_argument('--graph_hid', type=int, default=8, help='Number of hidden units')
parser.add_argument('--sigma', type=float, default=1.2, help='Sigma value for similarity matrix')
parser.add_argument('--adjacency_type', type=int, default=1, choices=[0,1,2], help='Type of adjacency matrix: '
                                                                '0 (fully connected graph),'
                                                                '1 (distances similarity matrix),'
                                                                '2 (knn similarity matrix).')
parser.add_argument('--top_k_neigh', type=int, default=None)

# GAT-specific
parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for graph attention network')
parser.add_argument('--alpha', type=float, default=0.2, help='Negative steep for the Leaky-ReLU activation')


# dirs
ROOT_DIR = pathlib.Path('.').absolute().parent.parent
BASE_DIR = ROOT_DIR / 'runs' / 'dagnet'
DATASET_DIR = BASE_DIR.parent.parent.absolute() / 'datasets' / 'sdd'

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
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             obs_goals, pred_goals_gt, non_linear_ped, loss_mask, seq_start_end) = data

            # goals one-hot encoding
            obs_goals_ohe = to_goals_one_hot(obs_goals, args.g_dim).cuda()

            # adj matrix for current batch
            if args.adjacency_type == 0:
                adj_out = compute_adjs(args, seq_start_end).cuda()
            elif args.adjacency_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj.detach().cpu(),
                                               pred_traj_gt.detach().cpu()).cuda()
            elif args.adjacency_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj.detach().cpu(),
                                              pred_traj_gt.detach().cpu()).cuda()

            kld, nll, ce, h = model(obs_traj, obs_traj_rel, obs_goals_ohe, seq_start_end, adj_out)

            ade, fde = [], []
            total_traj += obs_traj.shape[1]

            for _ in range(num_samples):
                samples_rel = model.sample(args.pred_len, h, obs_traj[-1], obs_goals_ohe[-1], seq_start_end)
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
    # current run main directory
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
    logging.info('Loading train/validation/test sets...')
    train_set, train_loader = data_loader(args, DATASET_DIR, 'train')
    valid_set, valid_loader = data_loader(args, DATASET_DIR, 'validation')
    test_set, test_loader = data_loader(args, DATASET_DIR, 'test')
    n_max_agents = max(train_set.__max_agents__(), valid_set.__max_agents__(), test_set.__max_agents__())

    model = DAGNet(args, n_max_agents).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logging.info('Evaluating...')

    ade, fde = evaluate(args, model, test_loader, args.num_samples)

    logging.info("### [DAG-NET] SDD RESULTS ###")
    logging.info("Pred Len: {}, ADE: {:.6f}, FDE: {:.6f}".format(args.pred_len, ade, fde))


if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)
