import numpy as np
import logging
import sys
import matplotlib.pyplot as plt

np.random.seed()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("gambler_problem")


class TerminationException:
    def __init__(self, reward):
        self.reward = reward


class GamblerPolicy:
    """
    States: s \in [1, 99]
    Actions: a \in [0, min(s, 100 - s)]
    """

    def __init__(self, p_head):
        self.states = range(0, 101)
        self._actions = lambda s: range(1, min(s, 100 - s) + 1)
        self._p_head = p_head

    def prob(self, state, action):
        """
        Given the state and action, return the probability of next step and corresponding rewards.
        There are two cases: head -> next_state = min(100, state + action) or tail -> next_state = max(0, state - action)
        :param state: current state
        :param action: current action
        :return: (next_state, probability, reward)
        """
        if state + action == 100:
            reward = 1
        else:
            reward = 0

        return [(state + action, self._p_head, reward), (state - action, 1 - self._p_head, 0)]

    def get_eligible_actions(self, state):
        return self._actions(state)

    @staticmethod
    def is_terminal_state(state):
        return state in [0, 100]


class PolicyIteration:
    def __init__(self, gambler_policy, threshold=1e-10, gamma=0.95):
        self._threshold = threshold
        self._gamma = gamma
        self._gambler_policy = gambler_policy
        self.state_value = {state: np.random.rand() for state in gambler_policy.states}
        # Terminal state has value 0
        self.state_value[0] = 0
        self.state_value[100] = 1
        self.policy = dict()

    def improve_policy(self):
        epoch = 0
        while True:
            delta = 0
            for state, value in self.state_value.items():
                if self._gambler_policy.is_terminal_state(state):
                    # Skip terminal states
                    continue

                new_value = 0
                for action in self._gambler_policy.get_eligible_actions(state):
                    proposed_value = sum([prob * (reward + self._gamma * self.state_value[next_state])
                                          for next_state, prob, reward in self._gambler_policy.prob(state, action)])
                    if proposed_value >= new_value:
                        # Update the policy, so that we can get optimal policy directly
                        new_value = proposed_value
                        self.policy[state] = action

                logger.debug("State {} changing from {} to {}".format(state, value, new_value))
                self.state_value[state] = new_value
                delta = max(delta, abs(new_value - value))

            if delta < self._threshold:
                break
            else:
                epoch += 1
                logger.debug("############# Epoch {} #############".format(epoch))

    def get_optimal_policy(self):
        self.improve_policy()
        return self.policy


if __name__ == "__main__":
    gambler_policy = GamblerPolicy(0.4)
    policy_iteration = PolicyIteration(gambler_policy, gamma=1)
    optimal_policy = policy_iteration.get_optimal_policy()
    state_value = policy_iteration.state_value

    logger.info(optimal_policy)

    fig, axs = plt.subplots(2)

    # Plot value estimates without last state (100)
    axs[0].plot(list(state_value.keys())[:-1], list(state_value.values())[:-1])
    axs[0].set_ylabel("Value estimates")

    axs[1].step(optimal_policy.keys(), optimal_policy.values())
    axs[1].set_ylabel("Final policy (stake")

    plt.show()
