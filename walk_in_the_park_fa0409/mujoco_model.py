# -*- coding: utf-8 -*-
import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

'''
Model of DDPG:  defines an Actor/policy network given obs as input,
                      & a Critic/value network given obs and action as input.
'''


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clip(min=eps)


class MujocoModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(MujocoModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)
        self.encoder_model1 = Encoder1(obs_dim)
        self.encoder_model2 = Encoder2(action_dim)

    def policy(self, obs, zs):
        return self.actor_model(obs, zs)

    def value(self, obs, action, zsa, zs):
        return self.critic_model(obs, action, zsa, zs)

    def encoder1(self, obs):
        return self.encoder_model1(obs)

    def encoder2(self, zs, action):
        return self.encoder_model2(zs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def get_encoder1_params(self):
        return self.encoder_model1.parameters()

    def get_encoder2_params(self):
        return self.encoder_model2.parameters()


class Encoder1(parl.Model):
    def __init__(self, obs_dim, zs_dim=256):
        super(Encoder1, self).__init__()

        # state encoder
        self.zs1 = nn.Linear(obs_dim, 256)
        self.zs2 = nn.Linear(256, 256)
        self.zs3 = nn.Linear(256, zs_dim)

    def forward(self, state):
        zs = F.elu(self.zs1(state))
        zs = F.elu(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs


class Encoder2(parl.Model):
    def __init__(self, action_dim, zs_dim=256):
        super(Encoder2, self).__init__()

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim * 4, 256)
        self.zsa2 = nn.Linear(256, 256)
        self.zsa3 = nn.Linear(256, zs_dim)

    def forward(self, zs, action):
        zsa = F.elu(self.zsa1(paddle.concat([zs, action], 1)))
        zsa = F.elu(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim, zs_dim=256, hdim=256):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, hdim)
        self.l2 = nn.Linear(zs_dim + hdim, hdim)
        self.l3 = nn.Linear(hdim, hdim)
        self.l4 = nn.Linear(hdim, action_dim)

    def forward(self, obs, zs):
        a = AvgL1Norm(self.l1(obs))
        a = paddle.concat([a, zs], 1)
        a = F.elu(self.l2(a))
        a = F.elu(self.l3(a))
        return paddle.tanh(self.l4(a))


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim, agent_num=None, zs_dim=256, hdim=256, dropout_rate=0.01):
        super(Critic, self).__init__()
        self.agent_num = agent_num

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.l1 = nn.Linear(obs_dim + action_dim * 4, hdim)
        self.l2 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.l3 = nn.Linear(hdim, hdim)
        self.l4 = nn.Linear(hdim, 1)

    def forward(self, obs, action, zsa, zs):
        sa = paddle.concat([obs, action], 1)
        embeddings = paddle.concat([zsa, zs], 1)

        q = AvgL1Norm(self.l1(sa))
        q = paddle.concat([q, embeddings], 1)
        # q = F.elu(self.l2(q))
        q = F.elu(self.l2(q))
        # q = self.dropout(F.elu(self.norm1(self.l2(q))))
        q = self.dropout(F.elu(self.norm2(self.l3(q))))
        q = self.l4(q)
        return q
