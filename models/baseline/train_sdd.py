import sys
import argparse
import pathlib
import torch
import random
import logging
import numpy as np
from math import ceil
from time import time
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

from models.baseline.model import VRNN
from models.baseline.data.loader_sdd import data_loader
from models.baseline.evaluate_sdd import evaluate
from models.utils.utils import relative_to_abs, average_displacement_error, final_displacement_error, plot_traj

parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--num_workers', default=0, type=int, required=False, help='Number of workers for loading data')
parser.add_argument('--obs_len', default=8, type=int, required=False, help='Timesteps of observation')
parser.add_argument('--pred_len', default=12, type=int, required=False, help='Timesteps of prediction')
parser.add_argument('--skip', default=1, type=int, required=False, help='Step for skipping frames')

# Optimization options
parser.add_argument('--learning_rate', default=1e-3, type=float, required=False, help='Initial learning rate')
parser.add_argument('--lr_scheduler', default=False, action='store_true', required=False, help='Learning rate scheduling')
parser.add_argument('--warmup', default=False, action='store_true', required=False, help='KLD warmup')
parser.add_argument('--batch_size', default=16, type=int, required=False, help='Batch size')
parser.add_argument('--num_epochs', default=500, type=int, required=False, help='Training epochs number')

# Model options
parser.add_argument('--clip', default=10, type=int, required=False, help='Gradient clipping')
parser.add_argument('--n_layers', default=2, type=int, required=False, help='Number of recurrent layers')
parser.add_argument('--x_dim', default=2, type=int, required=False, help='Dimension of the input of the single agent')
parser.add_argument('--h_dim', default=64, type=int, required=False, help='Dimension of the hidden layers')
parser.add_argument('--z_dim', default=16, type=int, required=False, help='Dimension of the latent variables')
parser.add_argument('--rnn_dim', default=64, type=int, required=False, help='Dimension of the recurrent layers')

# Miscellaneous
parser.add_argument('--seed', default=128, type=int, required=False, help='PyTorch random seed')
parser.add_argument('--print_every_batch', default=30, type=int, required=False, help='How many batches to print loss inside an epoch')
parser.add_argument('--save_every', default=50, type=int, required=False, help='How often save model checkpoint')
parser.add_argument('--eval_every', default=20, type=int, required=False, help='How often evaluate current model')
parser.add_argument('--num_samples', default=20, type=int, required=False, help='Number of samples for evaluation')
parser.add_argument('--run', required=True, type=str, help='Current run name')
parser.add_argument('--resume', default=False, action='store_true', help='Resume from last saved checkpoint')


# dirs
ROOT_DIR = pathlib.Path('.').absolute().parent.parent
BASE_DIR = ROOT_DIR / 'runs' / 'baseline'
DATASET_DIR = BASE_DIR.parent.parent.absolute() / 'datasets' / 'sdd'

# global vars
FORMAT = '[%(levelname)s | %(asctime)s]: %(message)s'
DATEFMT = '%Y/%m/%d %H:%M'
DEBUG = False


def save_checkpoint(fn, args, epoch, model, optimizer, lr_scheduler, best_ade, ade_list_val, fde_list_val,
                    ade_list_test, fde_list_test):
    torch.save({
        'args': args.__dict__,
        'epoch': epoch,
        'best_ade': best_ade,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'ade_list_val': ade_list_val,
        'fde_list_val': fde_list_val,
        'ade_list_test': ade_list_test,
        'fde_list_test': fde_list_test
    }, fn)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test(args, epch, test_loader, model, warmup, writer):
    test_loss = 0
    kld_loss = 0
    nll_loss = 0

    total_traj = 0
    ade = 0
    fde = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = [tensor.cuda() for tensor in data]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = data

            kld, nll, h = model(obs_traj_rel)
            loss = (warmup[epch - 1] * kld) + nll

            test_loss += loss.item()
            kld_loss += kld.item()
            nll_loss += nll.item()

            if batch_idx % args.print_every_batch == 0:
                writer.add_scalar('test/loss_items', loss.item(), epch) if not DEBUG else None

            # predict trajectories from latest h
            samples_rel = model.sample(args.pred_len, h)
            samples = relative_to_abs(samples_rel, obs_traj[-1])

            total_traj += samples.shape[1]  # num_seqs
            ade += average_displacement_error(samples, pred_traj_gt)
            fde += final_displacement_error(samples[-1, :, :], pred_traj_gt[-1, :, :])

        mean_loss = loss / len(test_loader.dataset)
        writer.add_scalar('test/Loss', mean_loss, epch) if not DEBUG else None
        mean_kld_loss = kld_loss / len(test_loader.dataset)
        mean_nll_loss = nll_loss / len(test_loader.dataset)
        writer.add_scalar('test/KLD_loss', mean_kld_loss, epch) if not DEBUG else None
        writer.add_scalar('test/NLL_loss', mean_nll_loss, epch) if not DEBUG else None

        # ADE
        ade_val = ade / (total_traj * args.pred_len)
        writer.add_scalar('test/ADE', ade_val, epch) if not DEBUG else None

        # FDE
        fde_val = fde / total_traj
        writer.add_scalar('test/FDE', fde_val, epch) if not DEBUG else None

        # plotting
        if DEBUG:
            obs = obs_traj.cpu().numpy()
            pred = samples.cpu().numpy()
            pred_gt = pred_traj_gt.cpu().numpy()
            plot_traj(obs, pred_gt, pred, seq_start_end, writer, epch)


def validate(args, epch, valid_loader, model, warmup, lr_scheduler, writer):
    test_loss = 0
    kld_loss = 0
    nll_loss = 0

    total_traj = 0
    ade = 0
    fde = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            data = [tensor.cuda() for tensor in data]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = data

            kld, nll, h = model(obs_traj_rel)
            loss = (warmup[epch-1] * kld) + nll

            test_loss += loss.item()
            kld_loss += kld.item()
            nll_loss += nll.item()

            if batch_idx % args.print_every_batch == 0:
                writer.add_scalar('validation/loss_items', loss.item(), epch) if not DEBUG else None

            # predict trajectories from latest h
            samples_rel = model.sample(args.pred_len, h)
            samples = relative_to_abs(samples_rel, obs_traj[-1])

            total_traj += samples.shape[1]  # num_seqs
            ade += average_displacement_error(samples, pred_traj_gt)
            fde += final_displacement_error(samples[-1, :, :], pred_traj_gt[-1, :, :])

        mean_loss = loss / len(valid_loader.dataset)
        writer.add_scalar('validation/Loss', mean_loss, epch) if not DEBUG else None
        mean_kld_loss = kld_loss / len(valid_loader.dataset)
        mean_nll_loss = nll_loss / len(valid_loader.dataset)
        writer.add_scalar('validation/KLD_loss', mean_kld_loss, epch) if not DEBUG else None
        writer.add_scalar('validation/NLL_loss', mean_nll_loss, epch) if not DEBUG else None

        # ADE
        ade_val = ade / (total_traj * args.pred_len)
        writer.add_scalar('validation/ADE', ade_val, epch) if not DEBUG else None

        # FDE
        fde_val = fde / total_traj
        writer.add_scalar('validation/FDE', fde_val, epch) if not DEBUG else None

        # eventually scheduler step
        if args.lr_scheduler and epch > 200:
            # let the metric settle for the first n epochs, then let the scheduler take care of the LR
            lr_scheduler.step(ade_val)

    return ade_val, fde_val


def train(args, epch, train_loader, model, warmup, optimizer, writer):
    train_loss = 0
    kld_loss = 0
    nll_loss = 0

    start = time()

    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = [tensor.cuda() for tensor in data]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end) = data

        seq_len = len(obs_traj) + len(pred_traj_gt)
        assert seq_len == args.obs_len + args.pred_len

        optimizer.zero_grad()
        kld, nll, _ = model(obs_traj_rel)
        loss = (warmup[epch-1] * kld) + nll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_loss += loss.item()
        kld_loss += kld.item()
        nll_loss += nll.item()

        if batch_idx % args.print_every_batch == 0:
            writer.add_scalar('train/loss_items', loss.item(), epch) if not DEBUG else None

    avg_loss = train_loss / len(train_loader.dataset)
    writer.add_scalar('train/Loss', avg_loss, epch) if not DEBUG else None
    avg_kld_loss = kld_loss / len(train_loader.dataset)
    avg_nll_loss = nll_loss / len(train_loader.dataset)
    writer.add_scalar('train/KLD_loss', avg_kld_loss, epch) if not DEBUG else None
    writer.add_scalar('train/NLL_loss', avg_nll_loss, epch) if not DEBUG else None

    end = time()
    elapsed = str(timedelta(seconds=(ceil(end - start))))
    logging.info('Epoch [{}], time elapsed: {}'.format(epch, elapsed))


def main(args):
    # current run main directory
    curr_run_dir = BASE_DIR / '{}'.format(args.run)
    curr_run_dir.mkdir(parents=True, exist_ok=True)

    # plain text logs
    log_full_path = curr_run_dir.absolute() / '{}.log'.format(args.run)
    logging.basicConfig(filename=log_full_path, level=logging.INFO, format=FORMAT, datefmt=DATEFMT)

    # data
    logging.info('Loading training/test sets...')
    _, train_loader = data_loader(args, DATASET_DIR, 'train')
    _, valid_loader = data_loader(args, DATASET_DIR, 'validation')
    _, test_loader = data_loader(args, DATASET_DIR, 'test')

    # saves directory
    save_dir = curr_run_dir / 'saves'
    save_best_dir = save_dir / 'best'
    eval_save_dir = save_dir / 'eval'
    save_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None
    save_best_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None
    eval_save_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None

    # maybe resume from last checkpoint
    if args.resume:
        save = list(pathlib.Path(save_dir).glob('*.pth'))[-1]
        checkpoint = torch.load(save)
        args.__dict__ = checkpoint['args']  # load original args from checkpoint
        args.__dict__['resume'] = True
        args.__dict__['learning_rate'] = args.learning_rate         # TODO: remove
        logging.info('Resuming...')

    # model, optim, lr scheduler
    model = VRNN(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=1e-2,
        patience=10,
        factor=5e-1,
        verbose=True
    )

    # maybe load weights/state from checkpoint
    model.load_state_dict(checkpoint['model_state_dict']) if args.resume else None
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if args.resume else None
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) if args.resume else None

    # starting epoch and best metric value seen so far
    start_epch = checkpoint['epoch'] if args.resume else 0
    best_ade = checkpoint['best_ade'] if args.resume else float('Inf')

    # tensorboard logs writer
    tb_dir = curr_run_dir / 'tb'
    tb_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None
    wr = (SummaryWriter(tb_dir, purge_step=start_epch + 1) if not DEBUG else None)

    # lists with metrics from periodic evaluations
    ade_list_test = checkpoint['ade_list_test'] if args.resume else []
    fde_list_test = checkpoint['fde_list_test'] if args.resume else []
    ade_list_val = checkpoint['ade_list_val'] if args.resume else []
    fde_list_val = checkpoint['fde_list_val'] if args.resume else []

    # logging hyperparams
    if not args.resume:
        args_dict = {}
        for arg in vars(args):
            args_dict.update({str(arg): str(getattr(args, arg))})
        wr.add_text('Parameters', str(args_dict), 0) if not DEBUG else None

    # KLD warmup
    wrmp_epochs = 50
    warmup = np.ones(args.num_epochs)
    warmup[:wrmp_epochs] = np.linspace(0, 1, num=wrmp_epochs) if args.warmup else warmup[:wrmp_epochs]

    for epoch in range(start_epch+1, args.num_epochs+1):
        train(args, epoch, train_loader, model, warmup, optimizer, wr)                      # train
        ade, _ = validate(args, epoch, valid_loader, model, warmup, lr_scheduler, wr)       # val
        test(args, epoch, test_loader, model, warmup, wr)                                   # test

        # check for new best
        if 0 <= ade.item() < best_ade:
            # save new best
            fn = '{}/checkpoint_epoch_{}.pth'.format(save_best_dir, str(epoch))
            best_ade = ade.item()
            for child in save_best_dir.glob('*'):
                if child.is_file():
                    child.unlink()  # delete previous best
            save_checkpoint(fn, args, epoch, model, optimizer, lr_scheduler, best_ade, ade_list_val, fde_list_val,
                            ade_list_test, fde_list_test)
            logging.info('Saved best checkpoint to ' + fn)

        # periodically save checkpoint
        if epoch % args.save_every == 0:
            fn = '{}/checkpoint_epoch_{}.pth'.format(save_dir, str(epoch))
            save_checkpoint(fn, args, epoch, model, optimizer, lr_scheduler, best_ade, ade_list_val, fde_list_val,
                            ade_list_test, fde_list_test)

        # periodically evaluate model
        if epoch % args.eval_every == 0:
            fn = '{}/checkpoint_epoch_{}.pth'.format(eval_save_dir, str(epoch))
            save_checkpoint(fn, args, epoch, model, optimizer, lr_scheduler, best_ade, ade_list_val, fde_list_val,
                            ade_list_test, fde_list_test)

            ade, fde = evaluate(args, model, test_loader, args.num_samples)
            ade_val, fde_val = evaluate(args, model, valid_loader, args.num_samples)

            ade_list_test.append((ade, epoch))
            fde_list_test.append((fde, epoch))
            ade_list_val.append((ade_val, epoch))
            fde_list_val.append((fde_val, epoch))

            logging.info('**** TEST **** ==> ADE: {}, FDE: {}'.format(ade, fde))
            logging.info('**** VALIDATION **** ==> ADE: {}, FDE: {}'.format(ade_val, fde_val))

        # print min metrics at the end of training
        if epoch == args.num_epochs:
            logging.info('*************Min metrics*************')
            ade_min_test, ade_min_ep_test = min(ade_list_test, key=lambda t: t[0])
            fde_min_test, fde_min_ep_test = min(fde_list_test, key=lambda t: t[0])
            logging.info('ADE Test: {} [epoch: {}]'.format(ade_min_test.item(), ade_min_ep_test))
            logging.info('FDE Test: {} [epoch: {}]'.format(fde_min_test.item(), fde_min_ep_test))

            ade_min_val, ade_min_ep_val = min(ade_list_val, key=lambda t: t[0])
            fde_min_val, fde_min_ep_val = min(fde_list_val, key=lambda t: t[0])
            logging.info('ADE Validation: {} [epoch: {}]'.format(ade_min_val.item(), ade_min_ep_val))
            logging.info('FDE Validation: {} [epoch: {}]'.format(fde_min_val.item(), fde_min_ep_val))

    wr.close()


if __name__== '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    DEBUG = (sys.gettrace() is not None)
    main(args)