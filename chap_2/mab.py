"""Multi-armed bandits based on chapter 2 in Reinforcement Learning book"""

import matplotlib.pyplot as plt

from framework.action_selectors import GreedyActionSelector
from framework.action_value_estimators import AverageRewardsEstimator, IncrementalRewardActionValueEstimator
from framework.rewards import NormalDistributionReward
from framework.runner import Runner


def run():
    fig, axs = plt.subplots(2)

    num_actions = 10

    epsilons = [0, 0.1, 0.01]
    colors = ['green', 'blue', 'red']
    steps = 1000
    epochs = 2000
    x = range(0, steps)

    for epsilon, color in zip(epsilons, colors):
        normal_dist_reward = NormalDistributionReward(num_actions)
        avg_reward_estimator = IncrementalRewardActionValueEstimator(num_actions)
        action_selector = GreedyActionSelector(avg_reward_estimator, epsilon)
        runner = Runner(normal_dist_reward, avg_reward_estimator, action_selector)
        total_avg_rewards, optimal_action_pct = runner.run_epochs(epochs, steps)
        axs[0].plot(x, total_avg_rewards, color=color, label="epsilon=" + str(epsilon))
        axs[1].plot(x, optimal_action_pct, color=color, label="epsilon=" + str(epsilon))

    axs[0].set_ylabel("Average reward")
    axs[1].set_ylabel('% Optimal action')
    axs[0].legend(loc='best')
    axs[1].legend(loc='best')
    plt.show()


if __name__ == '__main__':
    run()
