#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import paddle
import paddle.nn.functional as F
from copy import deepcopy
from parl.utils.utils import check_model_method
import numpy as np

__all__ = ['DDPG']
beta = 0.1
alpha = 1.0
num = 30
UTD = 10

class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None,
                 agent_index=None,
                 tag = None):
        """ DDPG algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_policy_noise = 0.2
        self.noise_clip = 0.5
        self.target_update_rate = 25
        self.tag1 = tag

        self.model = model
        self.target_model = deepcopy(self.model)
        self.target_model_extra = deepcopy(self.model)
        self.agent_index = agent_index
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())
        self.encoder1_optimizer = paddle.optimizer.Adam(
            learning_rate=3e-4, parameters=self.model.get_encoder1_params())  
        self.encoder2_optimizer = paddle.optimizer.Adam(
            learning_rate=3e-4, parameters=self.model.get_encoder2_params())      
        self.total_it = 0

    def predict(self, obs, agents):
        if self.tag1 == 1:
            self.target_model = deepcopy(self.model)
            self.target_model_extra = deepcopy(self.model)
            self.tag1 = 0
        zs = agents[0].alg.target_model.encoder1(obs)
        return self.model.policy(obs, zs)

    def learn(self, obs, action, reward, next_obs, terminal, agents,  total_steps):
        self.total_it = self.total_it + 1
        # Update Encoder
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal, agents)
        self._encoder_learn(obs, action, next_obs, agents)

        if self.total_it % UTD == 0: 
            actor_loss = self._actor_learn(obs, agents)
            self.sync_target1(total_steps)

        return critic_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal, agents):
        with paddle.no_grad():
            # Compute the target Q value
            # first compute next s'_A
            fixed_target_zs = agents[0].alg.target_model_extra.encoder1(next_obs)
            for i, agent in enumerate(agents):
                if i == 0:
                    action1 = agent.alg.target_model.policy(next_obs, fixed_target_zs)
                else:
                    action2 = agent.alg.target_model.policy(next_obs, fixed_target_zs)
                    action3 = paddle.concat([action1, action2], axis=-1)
            # action3 = np.array(action3)
            next_action = action3
            noise1 = (np.random.normal(0, self.target_policy_noise, size=6)).clip(-self.noise_clip, self.noise_clip)
            # 计算fixed_target_zsa
            next_action = (next_action + noise1).clip(-1,1)
            fixed_target_zsa = agents[0].alg.target_model_extra.encoder2(fixed_target_zs, next_action)
            next_obs = paddle.tile(next_obs, [num + 1, 1])
            next_action = self._expand(next_action)
            next_action = next_action.clip(-1.0, 1.0)
            fixed_target_zsa = paddle.tile(fixed_target_zsa, [num + 1, 1])
            fixed_target_zs = paddle.tile(fixed_target_zs, [num + 1, 1])
            target_next_P1 = agents[0].alg.target_model.value(next_obs, next_action, fixed_target_zsa, fixed_target_zs)
            target_next_P2 = agents[1].alg.target_model.value(next_obs, next_action, fixed_target_zsa, fixed_target_zs)
            target_next_P = paddle.minimum(target_next_P1, target_next_P2)
            target_next_P = paddle.reshape(target_next_P, shape=(num + 1, -1))
            p_next_s = self.mellow_max(target_next_P)
            p_next_s = paddle.reshape(p_next_s, shape=(-1, 1))

            # then compute current s_A
            target_obs = paddle.tile(obs, [num + 1, 1])
            current_action = action
            current_action = self._expand(current_action)
            current_action.clip(-1.0, 1.0)
            fixed_zs1 = agents[0].alg.target_model_extra.encoder1(target_obs)
            fixed_zsa1 = agents[0].alg.target_model_extra.encoder2(fixed_zs1, current_action)
            target_current_P1 = agents[0].alg.target_model.value(target_obs, current_action, fixed_zsa1, fixed_zs1)
            target_current_P2 = agents[1].alg.target_model.value(target_obs, current_action, fixed_zsa1, fixed_zs1)
            target_current_P = paddle.minimum(target_current_P1, target_current_P2)
            target_current_P = paddle.reshape(target_current_P, shape=(num + 1, -1))
            p_current_s = self.mellow_max(target_current_P)
            p_current_s = paddle.reshape(p_current_s, shape=(-1, 1))

            # finally compute current Q(s_a)
            fixed_zs2 = agents[0].alg.target_model_extra.encoder1(obs)
            fixed_zsa2 = agents[0].alg.target_model_extra.encoder2(fixed_zs2, action)
            target_current_Q1 = agents[0].alg.target_model.value(obs, action, fixed_zsa2, fixed_zs2)
            target_current_Q2 = agents[1].alg.target_model.value(obs, action, fixed_zsa2, fixed_zs2)
            target_current_Q = paddle.minimum(target_current_Q1, target_current_Q2)
            # target_current_Q = self.target_model.value(obs, action)

            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = (reward + ((1. - terminal) * self.gamma * p_next_s) +
                        alpha * (target_current_Q - p_current_s))


        fixed_zs3 = agents[0].alg.target_model.encoder1(obs)
        fixed_zsa3 = agents[0].alg.target_model.encoder2(fixed_zs3,action)
        # Get current Q estimate
        current_Q1 = agents[0].alg.model.value(obs, action, fixed_zsa3, fixed_zs3)
        current_Q2 = agents[1].alg.model.value(obs, action, fixed_zsa3, fixed_zs3)
        # current_Q = self.model.value(obs, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs, agents):
        fixed_zs = agents[0].alg.target_model.encoder1(obs)
        # Compute actor loss and Update the frozen target models
        for i, agent in enumerate(agents):
            if i == 0:
                action4 = agent.alg.model.policy(obs, fixed_zs)
            else:
                action5 = agent.alg.model.policy(obs, fixed_zs)
                action6 = paddle.concat([action4, action5], axis=-1)
        fixed_zsa = agents[0].alg.target_model.encoder2(fixed_zs, action6)
        actor_loss = -self.model.value(obs, action6, fixed_zsa, fixed_zs).mean()

        # Optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss
    
    def _encoder_learn(self, obs, action, next_obs, agents):
        with paddle.no_grad():
            next_zs = agents[0].alg.model.encoder1(next_obs)
        zs = agents[0].alg.model.encoder1(obs)
        pred_zs = agents[0].alg.model.encoder2(zs,action)
        encoder_loss = F.mse_loss(pred_zs,next_zs)
        agents[0].alg.encoder1_optimizer.clear_grad()
        encoder_loss.backward(retain_graph = True)
        agents[0].alg.encoder1_optimizer.step()
        agents[0].alg.encoder2_optimizer.clear_grad()
        encoder_loss.backward()
        agents[0].alg.encoder2_optimizer.step()

    """
    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
        self.model.sync_weights_to(self.target_model_extra, decay=decay)
    """
    def sync_target1(self, total_steps):
        # if total_steps % self.target_update_rate == 0:
        # self.target_model_extra = deepcopy(self.target_model)
        # self.target_model = deepcopy(self.model)
        if total_steps % self.target_update_rate == 0:
            # print("ddddd")
            self.target_model_extra = deepcopy(self.target_model)
            self.target_model = deepcopy(self.model)

    def _expand(self, action):

        action_batch = action.shape[0]
        tile_action = paddle.tile(action, [num, 1])
        '''
        mu, sigma = 0, 0.2
        lower, upper = mu - 3 * sigma, mu + 3 * sigma
        trunc = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        sample = trunc.rvs(num)
        '''
        sample = paddle.normal(mean=0, std=0.1, shape=[1, num])
        #sample = sample.clip(-0.5, 0.5)
        sample = paddle.to_tensor(sample)
        sample = paddle.reshape(sample, shape=(-1, 1))
        sample = paddle.tile(sample, [1, action_batch])
        sample = paddle.flatten(sample)
        sample = paddle.to_tensor(sample)
        sample = paddle.reshape(sample, shape=(-1, 1))
        sample = paddle.cast(sample, dtype=paddle.float32)
        expand_action = tile_action + sample
        action = paddle.concat(x=[action, expand_action], axis=0)
        return action

    # 解决了溢出问题
    def mellow_max(self, q_vals):
        max_q_vals = paddle.max(q_vals, axis=0, keepdim=True)
        q_vals = q_vals - max_q_vals
        e_beta_Q = paddle.exp(beta * q_vals)

        sum_e_beta_Q = paddle.sum(e_beta_Q, 0) / (num + 1)
        max_q_vals = paddle.squeeze(max_q_vals)
        log_sum_Q = paddle.log(sum_e_beta_Q) + max_q_vals * beta

        softmax_q_vals = log_sum_Q/beta

        return softmax_q_vals

    def get_q(self, obs, action):
        return self.model.value(obs, action)
