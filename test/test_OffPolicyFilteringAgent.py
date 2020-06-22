from unittest import TestCase
from torchagents.algorithms import OffPolicyFilteringAgent
import gym
import torch


class TestOffPolicyFilteringAgent(TestCase):
    def test_off_policy_filtering_agent(self):
        torch.random.manual_seed(1337)
        env = gym.make("CartPole-v0")
        state_shape = env.observation_space.shape
        num_actions = env.action_space.n
        batch_size = 16
        num_batches = 50
        ce_agent = OffPolicyFilteringAgent(state_shape=state_shape,
                                           num_actions=num_actions,
                                           max_episodes=batch_size,
                                           percentile_threshold=70.0)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        high_percentile_avg_reward = 0
        # env = gym.wrappers.Monitor(env, directory="mon", force=True)

        for i in range(num_batches):
            for j in range(batch_size):

                state = env.reset()
                state = torch.tensor([state], dtype=torch.float32, device=device)
                finished_episode = False

                while finished_episode is False:
                    action = ce_agent.get_action(state)
                    next_state, reward, finished_episode, _ = env.step(action.cpu().numpy())
                    next_state = torch.tensor([next_state],
                                              dtype=torch.float32, device=device)
                    reward = torch.tensor([reward], dtype=torch.float32, device=device)
                    ce_agent.update_current_episode(state, action, reward, next_state)
                    state = next_state

                ce_agent.finished_episode()
            ce_agent.train()
            high_percentile_avg_reward = ce_agent.get_avg_reward()
            print(high_percentile_avg_reward)
            ce_agent.reset_episodes()
        self.assertTrue(170 < high_percentile_avg_reward < 201)
