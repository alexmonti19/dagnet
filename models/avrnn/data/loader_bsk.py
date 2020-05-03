import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def _read_files(_path):
    sequences = np.load('{}/trajectories.npy'.format(_path))
    goals = np.load('{}/goals.npy'.format(_path))
    return sequences, goals


def seq_collate(data):
    (obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
     obs_goals, pred_goals) = zip(*data)

    batch_size = len(obs_traj)
    obs_seq_len, n_agents, features = obs_traj[0].shape
    pred_seq_len, _, _ = pred_traj[0].shape

    obs_traj = torch.cat(obs_traj, dim=1)
    pred_traj = torch.cat(pred_traj, dim=1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=1)
    pred_traj_rel = torch.cat(pred_traj_rel, dim=1)
    obs_goals = torch.cat(obs_goals, dim=1)
    pred_goals = torch.cat(pred_goals, dim=1)

    # fixed number of agent for every play -> we can manually build seq_start_end
    idxs = list(range(0, (batch_size * n_agents) + n_agents, n_agents))
    seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, obs_goals,
        pred_goals, torch.tensor(seq_start_end)
    ]
    return tuple(out)


class BasketDataset(Dataset):
    """Dataloder for the Basketball trajectories datasets"""

    def __init__(self, path, n_agents, obs_len=10, pred_len=40):
        super(BasketDataset, self).__init__()

        self.data_dir = path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents

        # 'trajectories' shape (seq_len, batch*n_agents, 2) -> 2 = coords (x,y) for the single player
        # 'goals' shape (seq_len, batch*n_agents)
        traj_abs, goals = _read_files(self.data_dir)

        assert traj_abs.shape[0] == self.seq_len and goals.shape[0] == self.seq_len
        #assert self.seq_len <= traj_abs.shape[0] and self.seq_len <= goals.shape[0]

        num_seqs = traj_abs.shape[1] // self.n_agents
        idxs = [idx for idx in range(0, (num_seqs * self.n_agents) + n_agents, n_agents)]
        seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

        self.num_samples = len(seq_start_end)

        traj_rel = np.zeros(traj_abs.shape)
        traj_rel[1:, :, :] = traj_abs[1:, :, :] - traj_abs[:-1, :, :]

        self.obs_traj = torch.from_numpy(traj_abs).type(torch.float)[:obs_len, :, :]
        self.obs_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[:obs_len, :, :]
        self.obs_goals = torch.from_numpy(goals).type(torch.float)[:obs_len, :]
        self.pred_traj = torch.from_numpy(traj_abs).type(torch.float)[obs_len:, :, :]
        self.pred_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[obs_len:, :, :]
        self.pred_goals = torch.from_numpy(goals).type(torch.float)[obs_len:, :]
        self.seq_start_end = seq_start_end

    def __len__(self):
        return self.num_samples

    def __max_agents__(self):
        # in bball the number of agents per scene is always the same
        return self.n_agents

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        out = [
            self.obs_traj[:, start:end, :], self.pred_traj[:, start:end, :],
            self.obs_traj_rel[:, start:end, :], self.pred_traj_rel[:, start:end, :],
            self.obs_goals[:, start:end], self.pred_goals[:, start:end]
        ]
        return out


###################################  CALLABLE LOADER  ###################################
def data_loader(args, dset_path, set, shuffle=True):

    if set not in ['train', 'validation', 'test']:
        raise(Exception("Dataset split not supported. Possible choices are: 'train', 'validation', 'test'"))

    n_agents = 10 if (args.players == 'all') else 5
    path = dset_path / args.players / set

    dataset = BasketDataset(
        path,
        n_agents=n_agents,
        obs_len=args.obs_len,
        pred_len=args.pred_len
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=seq_collate)
    return dataset, loader
