import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
from parl.env import ActionMappingWrapper, CompatWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from ddpg3 import DDPG
from parl.utils import CSVLogger
import paddle
import time


WARMUP_STEPS = 1e4 # change from 1e4 to 1e2
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise
UTD =10
RESTORE_TAG = 0 

paddle.device.set_device('gpu:5') # gpu 2 4 5

# Run episode for training
def run_train_episode(agents, env, rpm, total_steps):
    action_dim = env.action_space.shape[0]//2
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    total_steps1 = total_steps
    while not done:
        episode_steps += 1
        total_steps1 += 1
        action = []
        for agent in agents:
            if rpm.size() < WARMUP_STEPS:
                action.extend(np.random.uniform(-1, 1, size=action_dim))
            else:
                action.extend(agent.sample(obs, agents))

        action = np.array(action)

        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0


        # Store data in replay memory
        rpm.append(obs, action, reward, next_obs, terminal)
        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            for k in range(UTD):
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
                for i, agent in enumerate(agents):
                    agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                batch_terminal, agents, total_steps1)

    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agents, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = []
            for agent in agents:
                action.extend(agent.predict(obs,agents))
            action = np.array(action)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("------------------ MACDPP_FACTORIZATION ---------------------")
    logger.info('Env: {}, Seed: {}, name: {}'.format(args.env, args.seed, args.name))
    logger.info("---------------------------------------------")
    logger.info(paddle.device.get_device())
    logger.set_dir('./{}_{}_{}'.format(args.env, args.seed, args.name))
    csv_logger = CSVLogger('./{}_{}_{}/result.csv'.format(args.env, args.seed, args.name))

    env = gym.make(args.env)
    # Compatible for different versions of gym
    env = CompatWrapper(env)
    env = ActionMappingWrapper(env)

    env.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] // 2

    # Initialize model, algorithm, agent, replay_memory
    agents = []
    for i in range(2):
        model = MujocoModel(obs_dim, action_dim)
        algorithm = DDPG(
            model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, agent_index=i, tag = RESTORE_TAG)
        agent = MujocoAgent(algorithm, action_dim, expl_noise=EXPL_NOISE, agent_index=i)
        agents.append(agent)

    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim * 2)

    total_steps = 0
    test_flag = 0

    while total_steps < args.train_total_steps:

        episode_reward, episode_steps = run_train_episode(agents, env, rpm, total_steps)
        total_steps += episode_steps


        summary.add_scalar('train/episode_reward', episode_reward, total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agents, env, EVAL_EPISODES)
            csv_logger.log_dict({"reward": avg_reward})
            summary.add_scalar('eval/episode_reward', avg_reward, total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))

    # save the model and parameters of policy network for inference
    save_inference_path = './inference_model'
    input_shapes = [[None, env.observation_space.shape[0]]]
    input_dtypes = ['float32']
    agent.save_inference_model(save_inference_path, input_shapes, input_dtypes,
                               model.actor_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="HalfCheetah-v3", help='OpenAI gym environment name')
    parser.add_argument("--seed", default=8, type=int, help='Sets Gym seed')
    parser.add_argument("--name", default=1, type=int, help='folder name')
    parser.add_argument(
        "--train_total_steps",
        default=3e5,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
