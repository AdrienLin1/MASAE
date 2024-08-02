#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import shutil

# import numpy as np
import tqdm

import gym
import wandb
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from rl.agents import SACLearner
from rl.data import ReplayBuffer
from rl.evaluation import evaluate
from rl.wrappers import wrap_gym

import gym
import argparse
import numpy as np
# 替换使用jax模块
# import jax.numpy as np
from parl.utils import logger, summary, ReplayMemory
from parl.env import ActionMappingWrapper, CompatWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from ddpg3 import DDPG
from parl.utils import CSVLogger
import paddle
import time

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'A1Run-v0', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 66, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 1,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(2e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('wandb', False, 'Log wandb.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_float('action_filter_high_cut', None, 'Action filter high cut.')
flags.DEFINE_integer('action_history', 1, 'Action history.')
flags.DEFINE_integer('control_frequency', 20, 'Control frequency.')
flags.DEFINE_integer('utd_ratio', 1, 'Update to data ratio.')
flags.DEFINE_boolean('real_robot', False, 'Use real robot.')
config_flags.DEFINE_config_file(
    'config',
    'configs/droq_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

WARMUP_STEPS = 1e4 # change from 1e4 to 1e2
EVAL_EPISODES = 1
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256 # 100 -- 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
AGENT_NUM = 4 # 分治成N个智能体
UTD = 20

paddle.device.set_device('gpu')

def run_evaluate_episodes(agents, env, eval_episodes):
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs, done = env.reset(), False
        while not done:
            action = []
            for agent in agents:
                action.extend(agent.predict(obs, agents))
            action = np.array(action)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main(_):
    # wandb.init(project='a1')
    # wandb.config.update(FLAGS)
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    t88 = int(time.time())
    log_file_path = os.path.join(log_dir, f'log_{t88}.log')

    if FLAGS.real_robot:
        from real.envs.a1_env import A1Real
        env = A1Real(zero_action=np.asarray([0.05, 0.9, -1.8] * 4))
    else:
        from env_utils import make_mujoco_env
        env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)

    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     f'videos/train_{FLAGS.action_filter_high_cut}',
    #     episode_trigger=lambda x: True)
    env.seed(FLAGS.seed)

    if not FLAGS.real_robot:
        eval_env = make_mujoco_env(
            FLAGS.env_name,
            control_frequency=FLAGS.control_frequency,
            action_filter_high_cut=FLAGS.action_filter_high_cut,
            action_history=FLAGS.action_history)
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        # eval_env = gym.wrappers.RecordVideo(
        #     eval_env,
        #     f'videos/eval_{FLAGS.action_filter_high_cut}',
        #     episode_trigger=lambda x: True)
        eval_env.seed(FLAGS.seed + 42)

    # kwargs = dict(FLAGS.config)
    obs_dim = env.observation_space.shape[0] # 这里的分治方法，每个agent仍然接受的是global observation
    action_dim = env.action_space.shape[0] // AGENT_NUM

    agents = []
    for i in range(AGENT_NUM):
        model = MujocoModel(obs_dim, action_dim)
        algorithm = DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
        agent = MujocoAgent(algorithm, action_dim, expl_noise=EXPL_NOISE, agent_index=i)
        agents.append(agent)

    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim * AGENT_NUM)

    # agent = SACLearner.create(FLAGS.seed, env.observation_space,
    #                           env.action_space, **kwargs)

    chkpt_dir = 'saved/checkpoints'
    os.makedirs(chkpt_dir, exist_ok=True)
    buffer_dir = 'saved/buffers'

    last_checkpoint = checkpoints.latest_checkpoint(chkpt_dir)

    if last_checkpoint is None:
        start_i = 0
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     FLAGS.max_steps)
        replay_buffer.seed(FLAGS.seed)
    else:
        start_i = int(last_checkpoint.split('_')[-1])

        agent = checkpoints.restore_checkpoint(last_checkpoint, agent)

        with open(os.path.join(buffer_dir, f'buffer_{start_i}'), 'rb') as f:
            replay_buffer = pickle.load(f)

    obs, done = env.reset(), False
    r1 = 0
    total_steps = 0
    test_flag = 0
    for i in tqdm.tqdm(range(start_i, FLAGS.max_steps),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        action = []
        for agent in agents:
            if i < FLAGS.start_training:
                action.extend(np.random.uniform(-1, 1, size=action_dim))
                # action = env.action_space.sample()
            else:
                action.extend(agent.sample(obs, agents))
                # action, agent = agent.sample_actions(observation)
        action = np.array(action)
        # print(action)
        # print(type(action))
        next_obs, reward, done, info = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)
        r1 = r1+reward

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        # replay_buffer.insert(
        #     dict(observations=observation,
        #          actions=action,
        #          rewards=reward,
        #          masks=mask,
        #          dones=done,
        #          next_observations=next_observation))
        # observation = next_observation
        obs = next_obs

        if done:
            # print(r1)
            # r1 = 0
            obs, done = env.reset(), False
            for k, v in info['episode'].items():
                decode = {'r': 'return', 'l': 'length', 't': 'time'}
                # wandb.log({f'training/{decode[k]}': v}, step=i)

        # 保存模型
        # if i % 5000 == 0 :
        #     count1 = i // 5000
        #     model_dir = './log/' + str(t88) + '/saved_model'
        #     for d in range(4):
        #         model_name = '/agent_' + str(d)
        #         model_path = model_dir + str(count1)  # .pt 是常用的文件扩展名
        #         os.makedirs(model_path, exist_ok=True)
        #         model_path = model_path + model_name
        #         agents[d].save(model_path)
        #         # torch.save(agents[i].algorithm.model.state_dict(), model_path)
        # 更新
        if i >= FLAGS.start_training:
            for k in range(UTD):
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                    BATCH_SIZE)
                for h, agent in enumerate(agents):
                    agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                batch_terminal, agents, i)

            # batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            # agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            # if i % FLAGS.log_interval == 0:
                # for k, v in update_info.items():
                    # wandb.log({f'training/{k}': v}, step=i)

        if i % FLAGS.eval_interval == 0:
            z = i // 1000
            if not FLAGS.real_robot:
                avg_reward = run_evaluate_episodes(agents, env, EVAL_EPISODES)
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'Iteration {z}: Episode Reward = {avg_reward}\n')
                print(avg_reward)
                # eval_info = evaluate(agent,reset
                #                      eval_env,
                #                      num_episodes=FLAGS.eval_episodes)
                # for k, v in eval_info.items():
                #     print(k)
                    # wandb.log({f'evaluation/{k}': v}, step=i)

            # checkpoints.save_checkpoint(chkpt_dir,
            #                             agent,
            #                             step=i + 1,
            #                             keep=20,
            #                             overwrite=True)

            try:
                shutil.rmtree(buffer_dir)
            except:
                pass

            os.makedirs(buffer_dir, exist_ok=True)
            with open(os.path.join(buffer_dir, f'buffer_{i+1}'), 'wb') as f:
                pickle.dump(replay_buffer, f)


if __name__ == '__main__':
    app.run(main)
