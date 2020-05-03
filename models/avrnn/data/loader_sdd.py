import pathlib
import json
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, seq_start_end
    ]
    return tuple(out)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """ Real dataloder for the Stanford Drone Dataset"""

    def __init__(self, data_dir, obs_len, pred_len, set, skip, disable=False, threshold=0.002, min_ped=1):

        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        """

        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len

        num_peds_in_seq = []
        loss_mask_list = []
        non_linear_ped = []
        seq_list = []
        seq_list_rel = []

        all_files = np.load(data_dir, allow_pickle=True)
        for file in tqdm(all_files, desc='Loading {} set'.format(set), disable=disable):
            sequences_in_curr_file = []

            scene_name = file[0]
            data = file[1]

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                # List of list: contains all positions in each frame
                frame_data.append(data[frame == data[:, 0], :])
            # Num sequences to consider for a specified sequence length (seq_len = obs_len + pred_len)
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
            # Cycle for the length of sequences
            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                # All data for the current sequence: from the current index to current_index + sequence length
                curr_seq_data = np.concatenate(frame_data[idx:(idx + self.seq_len)], axis=0)
                # IDs of pedestrians in the current sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []
                # Cycle on pedestrians for the current sequence index
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # Current sequence for each pedestrian
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # Start frame for the current sequence of the current pedestrian reported to 0
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    # End frame for the current sequence of the current pedestrian:
                    # end of current pedestrian path in the current sequence.
                    # It can be sequence length if the pedestrian appears in all frame of the sequence
                    # or less if it disappears earlier.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # Exclude trajectories less then seq_len
                    if pad_end - pad_front != self.seq_len:
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        sequences_in_curr_file.append(curr_seq[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])

            if len(sequences_in_curr_file) > 0:
                all_traj = np.concatenate(sequences_in_curr_file, axis=0)

        self.num_seq = len(seq_list)
        self.seq_list = np.concatenate(seq_list, axis=0)
        self.seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(self.seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(self.seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

        # collect idxs to identificate the start-end indices (inside the batch dim) for single sequences
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [self.obs_traj[start:end, :], self.pred_traj[start:end, :],
               self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
               self.non_linear_ped[start:end], self.loss_mask[start:end, :],
        ]
        return out


###################################  CALLABLE LOADER  ###################################
def data_loader(args, dset_path, set, shuffle=True, disable=True, skip=1):

    if set not in ['train', 'validation', 'test']:
        raise(Exception("Dataset split not supported. Possible choices are: 'train', 'validation', 'test'"))

    path_to_npy_files = pathlib.Path(dset_path) / 'sdd_npy'
    full_path_to_set_file = '{}/{}.npy'.format(path_to_npy_files, set)

    dataset = TrajectoryDataset(
        full_path_to_set_file,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        set=set,
        skip=skip,
        disable=disable,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=seq_collate)
    return dataset, loader
