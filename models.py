import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, x_dim, u_dim, u_max, h_dim=256):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Linear(h_dim, u_dim),
            nn.Tanh()
        )

        self.u_max = u_max


    def forward(self, x):
        return self.u_max * self.layers(x)


class Critic(nn.Module):
    def __init__(self, x_dim, u_dim, h_dim=256):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(x_dim + u_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Linear(h_dim, 1)
        )

    def forward(self, x, u):
        x_u = torch.cat([x, u], 1)
        return self.layers(x_u)

class TD3():
    def __init__(
        self,
        x_dim,
        u_dim,
        u_max,
        h_dim=256,
        gamma=0.99,
        u_noise_std=0.2,
        u_noise_max=0.5,
        policy_update_period=2,
        replay_buffer=None):

        self.actor = Actor(x_dim, u_dim, u_max, h_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(x_dim, u_dim, h_dim)
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2 = Critic(x_dim, u_dim, h_dim)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_optim = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.u_max = u_max # max input
        self.u_noise_std = u_noise_std # std dev
        self.u_noise_max = u_noise_max # noise is clipped
        self.gamma = gamma # discount
        self.tau = .005 # exponential averaging constant
        self.policy_update_period = policy_update_period

        self.batch_size = 256
        self.replay_buffer = []

    def add_replay_buffer_sample(self, x, u, x_next, reward, done):
        x_rb, u_rb, x_next_rb, reward_rb, done_rb = self.replay_buffer
        self.replay_buffer = [
            torch.cat((x_rb, torch.from_numpy(x)), dim=1),
            torch.cat((u_rb, torch.from_numpy(u)), dim=1),
            torch.cat((x_next_rb, torch.from_numpy(x_next)), dim=1),
            torch.cat((reward_rb, torch.from_numpy(reward)), dim=1),
            torch.cat((done_rb, torch.from_numpy(done)), dim=1),
        ]


    def sample_replay_buffer(self):

        sample_inds = np.random.randint(0, len(self.replay_buffer), self.batch_size)
        get_rb_ind = lambda i: torch.cat([torch.from_numpy(self.replay_buffer[s][i]).view(1, -1)
            for s in sample_inds], dim=0).float().to(device)
        x = get_rb_ind(0)
        u = get_rb_ind(1)
        x_next = get_rb_ind(2)
        reward = torch.tensor([self.replay_buffer[s][3] for s in sample_inds]).float().view(-1, 1).to(device)
        done = torch.tensor([self.replay_buffer[s][4] for s in sample_inds]).float().view(-1, 1).to(device)
        return x, u, x_next, reward, done

    def critic_loss(self, batch):
        # sample replay_buffer
        x, u, x_next, reward, done = batch

        with torch.no_grad():
            u_noise = torch.normal(0, self.u_noise_std, size=u.shape).clip(-self.u_noise_max, self.u_noise_max)

            u_next = (self.actor_target(x_next) + u_noise).clip(-self.u_max, self.u_max)

            critic_target_q = reward + self.gamma * (1 - done) * \
                torch.min(self.critic_1_target(x_next, u_next), self.critic_2_target(x_next, u_next))

        loss_fn = torch.nn.MSELoss()
        critic_loss = loss_fn(self.critic_1(x, u), critic_target_q) + \
            loss_fn(self.critic_2(x, u), critic_target_q)

        return critic_loss

    def actor_loss(self, batch):

        x, u, x_next, reward, done = batch
        actor_loss = -self.critic_1(x, self.actor(x)).mean()
        return actor_loss
