import numpy as np
import torch
import torch.nn as nn

from models.dagnet.gat.gat_model import GAT
from models.dagnet.gcn.gcn_model import GCN
from models.utils.adjacency_matrix import adjs_fully_connected_pred, adjs_distance_sim_pred, adjs_knn_sim_pred
from models.utils.utils import sample_multinomial

class DAGNet (nn.Module):
    def __init__(self, args, n_max_agents):
        super(DAGNet, self).__init__()

        self.n_max_agents = n_max_agents
        self.n_layers = args.n_layers
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim
        self.d_dim = n_max_agents*2
        self.g_dim = args.g_dim
        self.rnn_dim = args.rnn_dim

        self.graph_model = args.graph_model
        self.graph_hid = args.graph_hid
        self.adjacency_type = args.adjacency_type
        self.top_k_neigh = args.top_k_neigh
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.n_heads = args.n_heads


        # goal generator
        self.dec_goal = nn.Sequential(
            nn.Linear(self.d_dim + self.g_dim + self.rnn_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.g_dim),
            nn.Softmax(dim=-1)
        )

        # goals graph
        if self.adjacency_type == 2 and self.top_k_neigh is None:
            raise Exception('Using KNN-similarity but top_k_neigh is not specified')

        if self.graph_model == 'gcn':
            self.graph_goals = GCN(self.g_dim, self.graph_hid, self.g_dim)
        elif self.graph_model == 'gat':
            assert self.n_heads is not None
            assert self.alpha is not None
            self.graph_goals = GAT(self.g_dim, self.graph_hid, self.g_dim, self.alpha, self.n_heads)

        # hiddens graph
        if self.adjacency_type == 2 and self.top_k_neigh is None:
            raise Exception('Using KNN-similarity but top_k_neigh is not specified')
        if self.graph_model == 'gcn':
            self.graph_hiddens = GCN(self.rnn_dim, self.graph_hid, self.rnn_dim)
        elif self.graph_model == 'gat':
            assert self.n_heads is not None
            assert self.alpha is not None
            self.graph_hiddens = GAT(self.rnn_dim, self.graph_hid, self.rnn_dim, self.alpha, self.n_heads)

        # interpolating original goals with refined goals from the first graph
        self.lg_goals = nn.Sequential(
            nn.Linear(self.g_dim + self.g_dim, self.g_dim),
            nn.Softmax(dim=-1)
        )

        # interpolating original hiddens with refined hiddens from the second graph
        self.lg_hiddens = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)

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
            nn.Linear(self.h_dim + self.g_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.enc_mean = nn.Linear(self.h_dim, self.z_dim)
        self.enc_logvar = nn.Linear(self.h_dim, self.z_dim)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(self.g_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.prior_mean = nn.Linear(self.h_dim, self.z_dim)
        self.prior_logvar = nn.Linear(self.h_dim, self.z_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(self.d_dim + self.g_dim + self.h_dim + self.rnn_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )
        self.dec_mean = nn.Linear(self.h_dim, self.x_dim)
        self.dec_logvar = nn.Linear(self.h_dim, self.x_dim)

        # recurrence
        self.rnn = nn.GRU(self.h_dim + self.h_dim, self.rnn_dim, self.n_layers)

    def _reparameterize(self, mean, log_var):
        logvar = torch.exp(log_var * 0.5).cuda()
        eps = torch.rand_like(logvar).cuda()
        return eps.mul(logvar).add(mean)

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


    def forward(self, traj, traj_rel, goals_ohe, seq_start_end, adj_out):
        timesteps, batch, features = traj.shape

        d = torch.zeros(timesteps, batch, features*self.n_max_agents).cuda()
        h = torch.zeros(self.n_layers, batch, self.rnn_dim).cuda()

        # an agent has to know all the xy abs positions of all the other agents in its sequence (for every timestep)
        for idx, (start,end) in enumerate(seq_start_end):
            n_agents = (end-start).item()
            d[:, start:end, :n_agents*2] = traj[:, start:end, :].reshape(timesteps, -1).unsqueeze(1).repeat(1,n_agents,1)

        KLD = torch.zeros(1).cuda()
        NLL = torch.zeros(1).cuda()
        cross_entropy = torch.zeros(1).cuda()

        for timestep in range(1, timesteps):
            x_t = traj_rel[timestep]
            d_t = d[timestep]
            g_t = goals_ohe[timestep]   # ground truth goal

            # refined goal must resemble real goal g_t
            dec_goal_t = self.dec_goal(torch.cat([d_t, h[-1], goals_ohe[timestep-1]], 1))
            g_graph = self.graph_goals(dec_goal_t, adj_out[timestep])  # graph refinement
            g_combined = self.lg_goals(torch.cat((dec_goal_t, g_graph), dim=-1))     # combination
            cross_entropy -= torch.sum(g_t * g_combined)

            # input feature extraction and encoding
            phi_x_t = self.phi_x(x_t)
            enc_t = self.enc(torch.cat([phi_x_t, g_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior
            prior_t = self.prior(torch.cat([g_t, h[-1]], 1))
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling from latent
            z_t = self._reparameterize(enc_mean_t, enc_logvar_t)

            # z_t feature extraction and decoding
            phi_z_t = self.phi_z(z_t)
            dec_t = self.dec(torch.cat([d_t, g_t, phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)

            # agent vrnn recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            KLD += self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            NLL += self._nll_gauss(dec_mean_t, dec_logvar_t, x_t)

            # hidden states refinement with graph
            h_graph = self.graph_hiddens(h[-1].clone(), adj_out[timestep])  # graph refinement
            h[-1] = self.lg_hiddens(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0) # combination

        return KLD, NLL, cross_entropy, h

    def sample(self, samples_seq_len, h, x_abs_start, g_start, seq_start_end):
        _, batch_size, _ = h.shape

        g_t = g_start   # at start, the previous goal is the last goal from GT observation
        x_t_abs = x_abs_start # at start, the curr abs pos of the agents come from the last abs pos from GT observations

        samples = torch.zeros(samples_seq_len, batch_size, self.x_dim).cuda()
        d = torch.zeros(samples_seq_len, batch_size, self.n_max_agents * self.x_dim).cuda()
        displacements = torch.zeros(samples_seq_len, batch_size, self.n_max_agents * 2).cuda()

        # at start, the disposition of the agents is composed by the last abs positions from GT obs
        for idx, (start,end) in enumerate(seq_start_end):
            n_agents = (end-start).item()
            d[0, start:end, :n_agents*2] = x_abs_start[start:end].reshape(-1).repeat(n_agents, 1)

        with torch.no_grad():
            for timestep in range(samples_seq_len):
                d_t = d[timestep]

                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end).cuda()
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu()).cuda()
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu()).cuda()

                # sampling agents' goals + graph refinement step
                dec_g = self.dec_goal(torch.cat([d_t, h[-1], g_t], 1))
                g_graph = self.graph_goals(dec_g, adj_pred)
                g_combined = self.lg_goals(torch.cat((dec_g, g_graph), dim=-1))
                g_t = sample_multinomial(torch.exp(g_combined))     # final predicted goal at current t

                # prior
                prior_t = self.prior(torch.cat([g_t, h[-1]], 1))
                prior_mean_t = self.prior_mean(prior_t)
                prior_logvar_t = self.prior_logvar(prior_t)

                # sampling from latent
                z_t = self._reparameterize(prior_mean_t, prior_logvar_t)

                # z_t feature extraction and decoding
                phi_z_t = self.phi_z(z_t)
                dec_t = self.dec(torch.cat([d_t, g_t, phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)
                dec_logvar_t = self.dec_logvar(dec_t)
                samples[timestep] = dec_mean_t

                # feature extraction for reconstructed samples
                phi_x_t = self.phi_x(dec_mean_t)

                # vrnn recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

                # graph refinement for agents' hiddens
                if self.adjacency_type == 0:
                    adj_pred = adjs_fully_connected_pred(seq_start_end).cuda()
                elif self.adjacency_type == 1:
                    adj_pred = adjs_distance_sim_pred(self.sigma, seq_start_end, x_t_abs.detach().cpu()).cuda()
                elif self.adjacency_type == 2:
                    adj_pred = adjs_knn_sim_pred(self.top_k_neigh, seq_start_end, x_t_abs.detach().cpu()).cuda()

                h_graph = self.graph_hiddens(h[-1].clone(), adj_pred)
                h[-1] = self.lg_hiddens(torch.cat((h_graph, h[-1]), dim=-1)).unsqueeze(0)

                # new abs pos
                x_t_abs = x_t_abs + dec_mean_t

                # disposition at t+1 is the current disposition d_t + the predicted displacements (dec_mean_t)
                if timestep != (samples_seq_len - 1):
                    for idx, (start, end) in enumerate(seq_start_end):
                        n_agents = (end - start).item()
                        displacements[timestep, start:end, :n_agents*2] = dec_mean_t[start:end].reshape(-1).repeat(n_agents, 1)
                    d[timestep + 1, :, :] = d_t[:, :] + displacements[timestep]

        return samples