# -*- coding: utf-8 -*-
import parl
import paddle
import numpy as np

sample_num = 50
tem = 1


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1, agent_index=None):
        assert isinstance(act_dim, int)
        super(MujocoAgent, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise
        self.agent_index = agent_index
        # self.alg.sync_target(decay=0)

    def sample(self, obs, agents):
        action_numpy = self.predict(obs, agents)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs, agents):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs, agents)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal, agents, total_steps):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss = self.alg.learn(obs, action, reward, next_obs,
                                     terminal, agents, total_steps)
        return critic_loss

    """
    def expand_action(self, action):
        action = paddle.reshape(action, shape=(-1, self.act_dim * 4))
        tile_action = paddle.tile(action, [sample_num, 1])
        '''
        mu, sigma = 0, 0.05
        lower, upper = mu - 3 * sigma, mu + 3 * sigma
        trunc = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        s_action = trunc.rvs(sample_num)
        '''
        s_action = paddle.normal(mean=0, std=0.1, shape=[1, sample_num])
        s_action = paddle.reshape(s_action, shape=(-1, 1))
        e_action = s_action + tile_action
        e_action = paddle.concat(x=[action, e_action], axis=0)
        return e_action
    """

    def softmax(self, x):
        e_x = paddle.exp(tem * x)
        sum_e_x = paddle.sum(e_x)
        dist_x = e_x / sum_e_x
        return dist_x