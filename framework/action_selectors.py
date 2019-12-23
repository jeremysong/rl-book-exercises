from abc import abstractmethod

import numpy as np
import random


class ActionSelector:
    """
    Select an action based on the estimated Q(a)
    """

    @abstractmethod
    def __init__(self, action_value_estimator):
        self._action_value_estimator = action_value_estimator

    @abstractmethod
    def select_action(self):
        pass


class GreedyActionSelector(ActionSelector):
    """
    Selects an action based mainly based on the greedy method. Randomness can be controlled by epsilon.
    :param epsilon: decides the probability of taking a random action
    """

    def __init__(self, action_value_estimator, epsilon):
        super().__init__(action_value_estimator)
        self._epsilon = epsilon

    def select_action(self):
        """
        Selects an action based mainly based on the greedy method. Randomness can be controlled by epsilon.
        :return: the selected action
        """
        estimated_q_a = self._action_value_estimator.get_estimated_q_a()

        if np.random.rand() < self._epsilon:
            chosen_action = random.choice(list(estimated_q_a.keys()))
        else:
            chosen_action = max(estimated_q_a, key=estimated_q_a.get)

        return chosen_action
