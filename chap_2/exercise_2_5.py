"""Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary 
problems. Use a modified version of the 10-armed testbed in which all the q⇤(a) start out equal and then take 
independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 
to all the q⇤(a) on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, 
incrementally computed, and another action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 
and longer runs, say of 10,000 steps. """

import matplotlib.pyplot as plt

from framework.action_selectors import GreedyActionSelector
from framework.action_value_estimators import AverageRewardsEstimator
from framework.rewards import RandomWalkActionReward
from framework.runner import Runner

import numpy as np


def run():
    fig, axs = plt.subplots(2)

    num_actions = 10
    init_q_a = np.random.rand()
    random_walk_action_reward = RandomWalkActionReward(init_q_a, num_actions)
    avg_reward_estimator = AverageRewardsEstimator(num_actions)

    epsilons = [0, 0.1, 0.01]
    colors = ['green', 'blue', 'red']
    steps = 100000
    epochs = 20
    x = range(0, steps)

    for epsilon, color in zip(epsilons, colors):
        action_selector = GreedyActionSelector(avg_reward_estimator, epsilon)
        runner = Runner(random_walk_action_reward, avg_reward_estimator, action_selector)
        total_avg_rewards, optimal_action_pct = runner.run_epochs(epochs, steps)
        axs[0].plot(x, total_avg_rewards, color=color, label="epsilon=" + str(epsilon))
        # axs[1].plot(x, optimal_action_pct, color=color, label="epsilon=" + str(epsilon))

    axs[0].set_ylabel("Average reward")
    # axs[1].set_ylabel('% Optimal action')
    axs[0].legend(loc='best')
    # axs[1].legend(loc='best')
    plt.show()


if __name__ == '__main__':
    run()
