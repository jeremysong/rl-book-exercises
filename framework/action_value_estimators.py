from abc import abstractmethod

import numpy as np


class ActionValueEstimator:
    """
    Estimates the action value
    """

    @abstractmethod
    def add_reward(self, action, reward):
        pass

    @abstractmethod
    def get_estimated_q_a(self):
        pass

    @abstractmethod
    def get_avg_reward(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class AverageRewardsEstimator(ActionValueEstimator):
    """
    Averages the rewards for each action.
    """

    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._action_occurrence_total_rewards = {}
        for action in range(0, num_actions):
            self._action_occurrence_total_rewards[action] = (0, 0.0)

    def add_reward(self, action, reward):
        """
        Updates the reward by counting the action has been taken and its reward
        """
        occurrence, total_rewards = self._action_occurrence_total_rewards[action]
        self._action_occurrence_total_rewards[action] = (occurrence + 1, total_rewards + reward)

    def get_estimated_q_a(self):
        """
        :return: estimated q(a) for each action by averaging all the rewards for each action
        """
        all_rewards = {}
        for action, occurrence_total_rewards in self._action_occurrence_total_rewards.items():
            if occurrence_total_rewards[0] == 0:
                all_rewards[action] = 0
            else:
                all_rewards[action] = occurrence_total_rewards[1] / occurrence_total_rewards[0]
        return all_rewards

    def get_avg_reward(self):
        """
        :return: estimated average rewards based on all the actions have been taken so far
        """
        total_occurrence = 0
        total_rewards = 0.0
        for occurrence, rewards in self._action_occurrence_total_rewards.values():
            total_occurrence += occurrence
            total_rewards += rewards
        return total_rewards / total_occurrence

    def reset(self):
        """
        Removes all estimated rewards
        """
        self._action_occurrence_total_rewards = {}
        for action in range(0, self._num_actions):
            self._action_occurrence_total_rewards[action] = (0, 0.0)


class IncrementalRewardActionValueEstimator(ActionValueEstimator):
    """
    Averages the rewards for each action, but each reward is incrementally computed
    """

    def __init__(self, num_actions):
        self._num_actions = num_actions
        self._action_occurrence_value = {}
        for action in range(0, num_actions):
            self._action_occurrence_value[action] = (0, 0.0)

    def add_reward(self, action, reward):
        """
        Updates the reward incrementally
        """
        occurrence, value = self._action_occurrence_value[action]
        occurrence += 1
        value = value + (reward - value) / occurrence
        self._action_occurrence_value[action] = (occurrence, value)

    def get_estimated_q_a(self):
        return {action: occurrence_value[1] for action, occurrence_value in self._action_occurrence_value.items()}

    def get_avg_reward(self):
        total_occurrence = 0
        total_rewards = 0.0
        for occurrence, value in self._action_occurrence_value.values():
            total_occurrence += occurrence
            total_rewards += value * occurrence
        return total_rewards / total_occurrence

    def reset(self):
        self._action_occurrence_value = {action: (0, 0.0) for action in self._action_occurrence_value}


class ConstantStepSizeActionValueEstimator(ActionValueEstimator):
    """
    Similar to **IncrementalRewardActionValueEstimator**, but the step size is fixed.
    """

    def __init__(self, num_actions, alpha):
        self._num_actions = num_actions
        self._alpha = alpha
        self._action_values = {}
        self._counter = 0
        self._total_reward = 0.0
        for action in range(0, num_actions):
            self._action_values[action] = 0.0

    def add_reward(self, action, reward):
        value = self._action_values[action]
        self._action_values[action] = value + self._alpha * (reward - value)
        self._total_reward += reward
        self._counter += 1

    def get_estimated_q_a(self):
        return self._action_values

    def get_avg_reward(self):
        return self._total_reward / self._counter

    def reset(self):
        self._action_values = {action: 0.0 for action in self._action_values}
        self._total_reward = 0.0
        self._counter = 0
