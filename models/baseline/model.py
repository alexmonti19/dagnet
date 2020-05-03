import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class VRNN(nn.Module):
    def __init__(self, args):
        super(VRNN, self).__init__()

        self.x_dim = args.x_dim
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.rnn_dim = args.rnn_dim
        self.n_layers = args.n_layers

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

    def _reparameterize (self, mean, log_var):
        logvar = torch.exp(log_var * 0.5).cuda()
        eps = torch.rand_like(logvar).cuda()
        return eps.mul(logvar).add(mean)

    def forward(self, x):
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
        h = Variable(torch.zeros(self.n_layers, batch, self.rnn_dim)).cuda()

        # we do not start from t=0 because with rel coordinates the first element is always (0,0)
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

    def sample(self, samples_seq_len, h):
        """
           Inputs:
           - h: last hidden from the network
           Outputs:
           - sample: tensor (pred_len, batch, 2) containing predicted future trajectories
        """
        _, batch, _ = h.shape

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

                samples[t] = dec_mean_t.data

                # feature extraction from reconstructed samples (~ 'x')
                phi_x_t = self.phi_x(dec_mean_t)

                # recurrence
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

        return samples


class VRNN_with_LSTM(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers):
        super(VRNN_with_LSTM, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_layers = n_layers

        # feature extraction
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU()
        )

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )

        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            # nn.Softplus()
        )

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            # nn.Softplus()
        )

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            # nn.Sigmoid()
        )
        self.dec_logvar = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            # nn.Softplus()
        )

        # recurrence
        self.rnn = nn.LSTM(h_dim + h_dim, h_dim, n_layers, False)

    def _reparameterize(self, mean, log_var):
        logvar = torch.exp(log_var * 0.5).cuda()
        eps = torch.rand_like(logvar).cuda()
        return eps.mul(logvar).add(mean)

    def forward(self, x):
        """
        Inputs:
        - x: tensor (obs_len, batch, 2) containing input observed data
        Outputs:
        - KLD: accumulated KLD values
        - NLL: accumulated NLL values
        - h: last hidden (-> useful for further sampling, if needed)
        """

        KLD = torch.zeros(1).cuda()
        NLL = torch.zeros(1).cuda()
        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).cuda()
        c = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).cuda()

        # we do not start from timestep 0 because in relative coordin. the first elem. is always (0,0)
        for t in range(1, x.size(0)):
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
            z_sampled = self._reparameterize(enc_mean_t, enc_logvar_t)
            phi_z_t = self.phi_z(z_sampled)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_logvar_t = self.dec_logvar(dec_t)

            # recurrence
            _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

            # losses
            KLD += self._kld(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            NLL += self._nll_gauss(dec_mean_t, dec_logvar_t, x[t])

        return KLD, NLL, h, c

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

    def sample(self, seq_len, h, c):
        """
           Inputs:
           - h: last hidden from the network
           - c: last cell state from LSTM
           Outputs:
           - sample: tensor (pred_len, batch, 2) containing predicted future trajectories
        """
        with torch.no_grad():
            sample = torch.zeros(seq_len, h.size(1), self.x_dim).cuda()

            for t in range(seq_len):
                # prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_logvar_t = self.prior_logvar(prior_t)

                # sampling and reparameterization
                z_t = self._reparameterize(prior_mean_t, prior_logvar_t)
                phi_z_t = self.phi_z(z_t)

                # decoder
                dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
                dec_mean_t = self.dec_mean(dec_t)

                sample[t] = dec_mean_t.data

                # recurrence
                phi_x_t = self.phi_x(dec_mean_t)
                _, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

        return sample