from unittest import TestCase
from torchagents.algorithms import CrossEntropyAgent
import gym
import torch


class TestCrossEntropy(TestCase):
    def test_cross_entropy(self):
        env = gym.make("CartPole-v0")
        state_shape = env.observation_space.shape
        num_actions = env.action_space.n
        num_episodes = 10
        num_batches = 50
        ce_agent = CrossEntropyAgent(state_shape=state_shape,
                                     num_actions=num_actions,
                                     max_episodes=num_episodes,
                                     percentile_threshold=0.7)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        avg_episode_reward = 0

        for _ in range(num_batches):
            for _ in range(num_episodes):
                state = env.reset()
                state = torch.tensor([state], dtype=torch.float32, device=device)
                finished_episode = False
                while finished_episode is False:
                    action = ce_agent.get_action(state)
                    next_state, reward, finished_episode, _ = env.step(action)
                    next_state = torch.tensor([next_state],
                                              dtype=torch.float32, device=device)
                    reward = torch.tensor([reward], dtype=torch.float32, device=device)
                    action = torch.tensor([action], dtype=torch.long, device=device)
                    ce_agent.update_current_episode(state, action, reward, next_state)
                ce_agent.finished_episode()
            ce_agent.train()
            avg_episode_reward = ce_agent.get_avg_reward()
            ce_agent.reset_episodes()
