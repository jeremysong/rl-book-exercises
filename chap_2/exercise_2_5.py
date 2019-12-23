"""Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary 
problems. Use a modified version of the 10-armed testbed in which all the q⇤(a) start out equal and then take 
independent random walks (say by adding a normally distributed increment with mean zero and standard deviation 0.01 
to all the q⇤(a) on each step). Prepare plots like Figure 2.2 for an action-value method using sample averages, 
incrementally computed, and another action-value method using a constant step-size parameter, ↵ = 0.1. Use " = 0.1 
and longer runs, say of 10,000 steps. """

import matplotlib.pyplot as plt

from framework.action_selectors import GreedyActionSelector
from framework.action_value_estimators import *
from framework.rewards import RandomWalkActionReward
from framework.runner import Runner

import numpy as np


def run():
    num_actions = 10
    init_q_a = np.random.rand()

    estimators = {
        'avg': AverageRewardsEstimator(num_actions),
        'incremental': IncrementalRewardActionValueEstimator(num_actions),
        'constant_step_size': ConstantStepSizeActionValueEstimator(num_actions, 0.1)
    }

    epsilon = 0.1
    colors = ['green', 'blue', 'red']
    steps = 10000
    epochs = 200
    x = range(0, steps)

    for (estimator_name, estimator), color in zip(estimators.items(), colors):
        random_walk_action_reward = RandomWalkActionReward(init_q_a, num_actions)
        action_selector = GreedyActionSelector(estimator, epsilon)
        runner = Runner(random_walk_action_reward, estimator, action_selector)
        total_avg_rewards, optimal_action_pct = runner.run_epochs(epochs, steps, meter=20)
        plt.plot(x, total_avg_rewards, color=color, label="estimator=" + estimator_name)

    plt.ylabel("Average reward")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    run()
