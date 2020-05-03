import argparse
import numpy as np
from math import ceil
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()

# args
parser.add_argument('--players', choices=['atk', 'def', 'all'], required=True, type=str, help='Which players')
parser.add_argument('--train_percentage', default=60, type=int, help='Percentage of the training split')
parser.add_argument('--validation_percentage', default=20, type=int, help='Percentage of the validation split')
parser.add_argument('--speed_threshold', default=0.5, type=float, help='Speed to determine if a player is moving or stationary')

# goals grid (the court is 47ft x 50 ft)
CELL_SIZE = 5   # 5ft x 5ft
N_CELLS_X = 9   # how many cells along X
N_CELLS_Y = 10  # how many cells along Y

# dirs
DATASETS_DIR = Path('.').absolute().parent
BASKET_DIR = DATASETS_DIR.absolute() / 'basket'


def clamp(val, lower, upper):
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


def get_goal_idx(position):
    eps = 1e-4
    x = clamp(position[0], 0, N_CELLS_X*CELL_SIZE-eps)
    y = clamp(position[1], 0, N_CELLS_Y*CELL_SIZE-eps)

    goal_idx_x = int(x/CELL_SIZE)
    goals_idx_y = int(y/CELL_SIZE)
    return goal_idx_x*N_CELLS_Y + goals_idx_y


def goals_stationary(args, agent_sequence):
    velocity = agent_sequence[1:, :] - agent_sequence[:-1, :]
    speed = np.linalg.norm(velocity, axis=-1)
    stationary = speed < args.speed_threshold
    stationary = np.append(stationary, True)

    timesteps = len(agent_sequence)
    goals = np.zeros(timesteps)

    for t in reversed(range(timesteps)):
        if t == (timesteps-1):
            # the cell occupied in last timestep is always a goal (-> usually the real final goal?)
            goals[t] = get_goal_idx(agent_sequence[t])
        elif not stationary[t + 1] and stationary[t]:
            # from moving to stationary: new goal
            goals[t] = get_goal_idx(agent_sequence[t])
        else:
            goals[t] = goals[t + 1]

    return goals


def compute_goals(args, paths):
    print('Computing goals...')

    for split in ['train', 'validation', 'test']:
        filename = '{}/trajectories.npy'.format(paths[split])
        data = np.load(filename)
        seqs, seq_len, _ = data.shape
        goals = np.zeros((seqs, seq_len, args.n_agents))

        for sequence in tqdm(range(seqs), desc='Processing {}'.format(split)):
            for agent in range(args.n_agents):
                goals[sequence, :, agent] = goals_stationary(args, data[sequence, :, 2*agent:2*agent+2])

        save_filename = '{}/goals.npy'.format(paths[split])
        np.save(save_filename, goals)


def paths(args):
    print('Creating directories')

    train_dir = BASKET_DIR.absolute() / args.players / 'train'
    valid_dir = BASKET_DIR.absolute() / args.players / 'validation'
    test_dir = BASKET_DIR.absolute() / args.players / 'test'

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    return {'train': train_dir, 'validation': valid_dir, 'test': test_dir}


def split_and_preprocess(args, paths):
    print('Splitting the dataset...')

    orig_dset_path = BASKET_DIR / 'all_data.npy'
    dataset = np.load(orig_dset_path, allow_pickle=True)

    # filter players
    if args.players == 'atk':
        dataset = dataset[:, :, 1:6, :]    # no ball, only attacking team
    elif args.players == 'def':
        dataset = dataset[:, :, 6:, :]  # no ball, only defense team
    else:
        dataset = dataset[:, :, 1:, :]  # no ball, only attacking team

    # preprocess
    dataset = np.delete(dataset, [2, 3], axis=3)    # keep only xy feature columns
    dataset[:, :, :, 0] = dataset[:, :, :, 0] - 45.0  # X coord: from 45<->90 to 0<->45
    seqs, seq_len, agents, features = dataset.shape
    dataset = dataset.reshape((seqs, seq_len, -1))    # (seqs, seq_len, player, xy) -> (seqs, seq_len, all_players_xy)

    # save splits
    training_split_idx = ceil((float(seqs) / 100) * args.train_percentage)
    validation_split_idx = ceil((float(seqs) / 100) * (args.validation_percentage + args.train_percentage))
    np.save(paths['train'] / 'trajectories.npy', dataset[:training_split_idx, :, :])
    np.save(paths['validation'] / 'trajectories.npy', dataset[training_split_idx:validation_split_idx, :, :])
    np.save(paths['test'] / 'trajectories.npy', dataset[validation_split_idx:, :, :])


if __name__ == '__main__':
    args = parser.parse_args()
    args.__dict__['n_agents'] = 10 if args.players == 'all' else 5

    assert args.train_percentage + args.validation_percentage <= 99

    pths = paths(args)
    split_and_preprocess(args, pths)
    compute_goals(args, pths)
    print('All done.')
