import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .gat.gat_model import GAT
from .gcn.gcn_model import GCN
from ..utils.adjacency_matrix import adjs_distance_sim_pred, adjs_knn_sim_pred, adjs_fully_connected_pred


class AVRNN(nn.Module):
    def __init__(self, args):
        super(AVRNN, self).__init__()

        self.x_dim = args.x_dim
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.rnn_dim = args.rnn_dim
        self.n_layers = args.n_layers

        self.graph_model = args.graph_model
        self.adjacency_type = args.adjacency_type
        self.top_k_neigh = args.top_k_neigh
        self.sigma = args.sigma
        self.graph_hid = args.graph_hid
        self.alpha = args.alpha
        self.n_heads = args.n_heads

        # feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_logvar = nn.Linear(self.h_dim, self.z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.dec_mean = nn.Linear(self.h_dim, self.x_dim)
        self.dec_logvar = nn.Linear(self.h_dim, self.x_dim)

        # recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.rnn_dim, self.n_layers)

        # graph
        if self.graph_model == 'gcn':
            self.graph = GCN(self.rnn_dim, self.graph_hid, self.rnn_dim)
        elif self.graph_model == 'gat':
            self.graph = GAT(self.rnn_dim, self.graph_hid, self.rnn_dim, self.alpha, self.n_heads)

        # interpolating original hiddens with hiddens from the graph
        self.lg = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)


    def _reparameterize (self, mean, log_var):
        logvar = torch.exp(log_var * 0.5).cuda()
        eps = torch.rand_like(logvar).cuda()
        return eps.mul(logvar).add(mean)


    def forward(self, x, adj):
        """
        Inputs:
        - x: tensor (obs_len, batch, 2) containing input observed data
        Outputs:
        - KLD: accumulated KLD values
        - NLL: accumulated NLL values
        - h: last hidden (-> useful for further sampling, if needed)
        """

        timesteps, batch, features = x.shape

        KLD = torch.zeros(1).cuda()
        NLL = torch.zeros(1).cuda()

        h = Variable(torch.zeros(self.n_layers, batch, self.rnn_dim), requires_grad=True).cuda()

        # we do not start from t=0 ecause with relative coords the first elem is always (0,0)
        for t in range(1, timesteps):
            # encoder
            phi_x_t = self.phi_x(x[t])
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sample from latent
            z_t = self._reparameterize(enc_mean_t, enc_logvar_t)

            # z_t feature extraction and decoding
            phi_z_t = self.phi_z(z_t)
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # hiddens graph refinement
            h_g = self.graph(h[-1].clone(), adj[t].cuda())
            h[-1] = self.lg(torch.cat((h_g, h[-1]), dim=1))

            # losses
            KLD += self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            NLL += self._nll_gauss(dec_mean_t, dec_logvar_t, x[t])

        return KLD, NLL, h


    def _kld(self, mean_enc, logvar_enc, mean_prior, logvar_prior):

        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) / (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3

        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean, logvar, x):
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))

        return nll

    def sample(self, samples_seq_len, h, x_abs_start, seq_start_end):
        """
           Inputs:
           - h: last hidden from the network
           Outputs:
           - sample: tensor (pred_len, batch, 2) containing predicted future trajectories
        """
        _, batch, _ = h.shape
        x_t_abs = x_abs_start   # at start, the curr abs pos of the agents come from the last abs pos from GT observations

        with torch.no_grad():
            samples = torch.zeros(samples_seq_len, batch, self.x_dim).cuda()

            for t in range(samples_seq_len):
                # prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_logvar_t = self.prior_logvar(prior_t)

                # sampling from latent
                z_t = self._reparameterize(prior_mean_t, prior_logvar_t)

                # z_t feature extraction and decoding
                phi_z_t = self.phi_z(z_t)
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_logvar_t = self.dec_logvar(dec_t)

                # feature extraction from reconstructed samples (~ 'x')
                phi_x_t = self.phi_x(dec_mean_t)

                # recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                # save predicted displacements
                samples[t] = dec_mean_t.data

                # new abs pos
                x_t_abs = x_t_abs + dec_mean_t

                # adjacencies
                if self.adjacency_type == 0:
                    adj_pred_t = adjs_fully_connected_pred(seq_start_end)
                elif self.adjacency_type == 1:
                    adj_pred_t = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu()).cuda()
                elif self.adjacency_type == 2:
                    adj_pred_t = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu()).cuda()

                # graph refinement and interpolation with vanilla hidden
                h_g = self.graph(h[-1].clone(), adj_pred_t)
                h[-1] = self.lg(torch.cat((h_g, h[-1]), dim=1))

        return samples