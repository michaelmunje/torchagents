import numpy as np
import gym

class Gym():
    n_steps = 0
    n_dones = 0

    def __init__(self, env_name, render=False):
        self.env = gym.make(env_name)
        self._render = render
        if self.is_discrete_action():
            self._action_space = tuple([self.env.action_space.n])
        else:
            self._action_space = self.env.action_space.shape
        self._state_space = self.env.observation_space.shape

    def step(self, action):
        self.n_steps += 1

        self._raw_action = action
        self._action = self._clean_actions(action)

        self._state, self._reward, self._done, _ = self.env.step(self._action)

        if self._done:
            self.n_dones += 1

        if self._render:
            self.env.render()

        return self._state, self._reward, self._done

    def _clean_actions(self, action):
        if self.is_discrete_action():
            return np.argmax(action)
        else:
            return np.array(action)

    def is_discrete_action(self):
        return self.env.action_space.shape == ()

    def get_last_action(self):
        return self._action

    def get_state(self):
        return self._state

    def get_reward(self):
        return self._reward

    def is_done(self):
        return self._done

    def get_action_space(self):
        return self._action_space

    def get_state_space(self):
        return self._state_space

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()