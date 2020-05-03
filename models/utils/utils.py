import torch
import matplotlib.pyplot as plt
import random
import numpy as np


######################### MISCELLANEOUS ############################
def relative_to_abs(traj_rel, start_pos):
    """
    Inputs:
    - rel_traj: tensor of shape (seq_len, batch, 2), trajectory composed by displacements
    - start_pos: tensor of shape (batch, 2), initial position
    Outputs:
    - input tensor (seq_len, batch, 2) filled with absolute coords
    """
    rel_traj = traj_rel.permute(1, 0, 2)        # (seq_len, batch, 2) -> (batch, seq_len, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos

    return abs_traj.permute(1, 0, 2)


def plot_traj(observed, predicted_gt, predicted, seq_start_end, writer, epch):
    """
    Inputs:
    - predicted: tensor (pred_len, batch, 2) with predicted trajectories
    - observed: tensor (obs_len, batch, 2) with observed trajectories
    - predicted_gt: tensor (pred_len, batch, 2) with predicted trajectories ground truth
    - seq_start_end: tensor (num_seq, 2) with temporal sequences start and end
    - writer: Tensorboard writer
    - epch: current epoch
    """

    idx = random.randrange(0, len(seq_start_end))    # print only one (random) sequence in the batch
    (start, end) = seq_start_end[idx]
    pred = predicted[:, start:end, :]
    obs = observed[:, start:end, :]
    pred_gt = predicted_gt[:, start:end, :]

    fig = plt.figure()

    for idx in range(pred.shape[1]):
        # draw observed
        plt.plot(obs[:, idx, 0], obs[:, idx, 1], color='green')

        # draw connection between last observed point and first predicted gt point
        x1, x2 = obs[-1, idx, 0], pred_gt[0, idx, 0]
        y1, y2 = obs[-1, idx, 1], pred_gt[0, idx, 1]
        plt.plot([x1, x2], [y1, y2], color='blue')

        # draw predicted gt
        plt.plot(pred_gt[:, idx, 0], pred_gt[:, idx, 1], color='blue')

        # draw connection between last observed point and first predicted point
        x1, x2 = obs[-1, idx, 0], pred[0, idx, 0]
        y1, y2 = obs[-1, idx, 1], pred[0, idx, 1]
        plt.plot([x1, x2], [y1, y2], color='red')

        # draw predicted
        plt.plot(pred[:, idx, 0], pred[:, idx, 1], color='red')

    writer.add_figure('Generations', fig, epch)
    fig.clf()


def std(vector, mean):
    sum = torch.zeros(mean.shape).cuda()
    for el in vector:
        sum += el - mean
    return torch.sqrt(torch.abs(sum) / len(vector))



######################### LOSSES ############################
def average_displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectories.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth predictions.
    Output:
    - error: total sum of displacement errors across all sequences inside the batch
    """

    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    loss = torch.sqrt(torch.sum(loss, dim=2))
    loss = torch.sum(loss, dim=1)

    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted final positions.
    - pred_pos_gt: Tensor of shape (batch, 2). Ground truth final positions.
    Output:
    - error: total sum of fde for all the sequences inside the batch
    """

    loss = (pred_pos - pred_pos_gt) ** 2
    loss = torch.sqrt(torch.sum(loss, dim=1))

    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='sum'):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    :param mode: Can be one of sum, average or raw
    :return: l2 loss depending on mode
    """

    loss = (loss_mask.cuda().unsqueeze(dim=2) * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def linear_velocity_acceleration(sequence, mode='mean', seconds_between_frames=0.2):
    seq_len, batch, features = sequence.shape

    velocity_x = torch.zeros((seq_len, batch)).cuda()
    velocity_y = torch.zeros((seq_len, batch)).cuda()
    for ped in range(batch):
        velocity_x[1:, ped] = (sequence[1:, ped, 0] - sequence[:-1, ped, 0]) / seconds_between_frames     # space / time
        velocity_y[1:, ped] = (sequence[1:, ped, 1] - sequence[:-1, ped, 1]) / seconds_between_frames     # space / time
    velocity = torch.sqrt(torch.pow(velocity_x, 2) + torch.pow(velocity_y, 2))

    acceleration_x = torch.zeros((seq_len, batch)).cuda()
    acceleration_y = torch.zeros((seq_len, batch)).cuda()
    acceleration = torch.zeros((seq_len, batch)).cuda()
    acceleration_x[2:, :] = (velocity_x[2:, :] - velocity_x[1:-1, :]) / seconds_between_frames
    acceleration_y[2:, :] = (velocity_y[2:, :] - velocity_y[1:-1, :]) / seconds_between_frames
    acceleration_comp = torch.sqrt(torch.pow(acceleration_x, 2) + torch.pow(acceleration_y, 2))
    acceleration[2:, :] = torch.abs((velocity[2:, :] - velocity[1:-1, :]) / seconds_between_frames)

    # Linear velocity means
    mean_velocity_x = velocity_x[1:].mean(dim=0)
    mean_velocity_y = velocity_y[1:].mean(dim=0)
    mean_velocity = velocity[1:].mean(dim=0)

    # Linear acceleration means
    mean_acceleration_x = acceleration_x[2:].mean(dim=0)
    mean_acceleration_y = acceleration_y[2:].mean(dim=0)
    mean_acceleration = acceleration[2:].mean(dim=0)

    # Linear velocity standard deviations
    std_velocity_x = std(velocity_x, mean_velocity_x)
    std_velocity_y = std(velocity_y, mean_velocity_y)
    sum_velocity = torch.sqrt(torch.pow(std_velocity_x, 2) + torch.pow(std_velocity_y, 2))
    std_velocity = torch.sqrt(torch.abs(sum_velocity) / len(sequence))

    # Linear acceleration standard deviations
    std_acceleration_x = std(mean_acceleration_x, mean_acceleration_x)
    std_acceleration_y = std(mean_acceleration_y, mean_acceleration_y)
    sum_acceleration = torch.sqrt(torch.pow(std_acceleration_x, 2) + torch.pow(std_acceleration_y, 2))
    std_acceleration = torch.sqrt(torch.abs(sum_acceleration) / len(sequence))

    if mode == 'raw':
        return mean_velocity, mean_acceleration, std_velocity_x, std_velocity_y
    elif mode == 'mean':
        # mode mean
        return mean_velocity.mean(), mean_acceleration.mean(), std_velocity.mean(), \
                std_acceleration.mean()
    else:
        raise(Exception("linear_velocity_acceleration(): wrong mode selected. Choices are ['raw', 'mean'].\n"))


######################### GOALS UTILITIES ############################
def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long()
    dims.append(N)
    ret = (torch.zeros(dims)).cuda()
    ret.scatter_(-1, inds, 1)
    return ret


def to_goals_one_hot(original_goal, ohe_dim):
    return one_hot_encode(original_goal[:, :].data, ohe_dim)


def divide_by_agents(data, n_agents):
    x = data[:, :, :2 * n_agents].clone()
    return x.view(x.size(0), x.size(1), n_agents, -1).transpose(1, 2)


def sample_multinomial(probs):
    """ Each element of probs tensor [shape = (batch, g_dim)] has 'g_dim' probabilities (one for each grid cell),
    i.e. is a row containing a probability distribution for the goal. We sample n (=batch) indices (one for each row)
    from these distributions, and covert it to a 1-hot encoding. """

    inds = torch.multinomial(probs, 1).data.long().squeeze()
    ret = one_hot_encode(inds, probs.size(-1))
    return ret


def get_goal(position, min_x, max_x, min_y, max_y, n_cells_x, n_cells_y):
    """Divides the scene rectangle into a grid of cells and finds the cell idx where the current position falls"""
    x = position[0]
    y = position[1]

    x_steps = np.linspace(min_x, max_x, num=n_cells_x+1)
    y_steps = np.linspace(min_y, max_y, num=n_cells_y+1)

    goal_x_idx = np.max(np.where(x_steps <= x)[0])
    goal_y_idx = np.max(np.where(y_steps <= y)[0])
    return (goal_x_idx * n_cells_y) + goal_y_idx


def compute_goals_fixed(tj, min_x, max_x, min_y, max_y, n_cells_x, n_cells_y, window=1):
    """Computes goals using the position at the end of each window."""
    timesteps = len(tj)
    goals = np.zeros(timesteps)
    for t in reversed(range(timesteps)):
        if (t + 1) % window == 0 or (t + 1) == T:
            goals[t] = get_goal(tj[t], min_x, max_x, min_y, max_y, n_cells_x, n_cells_y)
        else:
            goals[t] = goals[t + 1]

    return goals


######################### MATRIX UTILITIES #############################
# FROM:  https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1).cuda()
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])


def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])


def block_diag_irregular(matrices):
    matrices = [permute2st(m, 2) for m in matrices]

    ns = torch.LongTensor([m.shape[0] for m in matrices])
    n = torch.sum(ns)
    batch_shape = matrices[0].shape[2:]

    v = torch.zeros(torch.Size([n, n]) + batch_shape)
    for ii, m1 in enumerate(matrices):
        st = torch.sum(ns[:ii])
        en = torch.sum(ns[:(ii + 1)])
        v[st:en, st:en] = m1
    return permute2en(v, 2)