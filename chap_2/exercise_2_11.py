"""Make a figure analogous to Figure 2.6 for the nonstationary case outlined in Exercise 2.5. Include the 
constant-step-size "-greedy algorithm with ↵=0.1. Use runs of 200,000 steps and, as a performance measure for each 
algorithm and parameter setting, use the average reward over the last 100,000 steps. """

import matplotlib.pyplot as plt

from framework.action_selectors import GreedyActionSelector
from framework.action_value_estimators import *
from framework.rewards import RandomWalkActionReward
from framework.runner import Runner


def epsilon_parameter_generator(init=1 / 128, ratio=2):
    element = init
    while True:
        yield element
        element *= ratio


def run():
    num_actions = 10
    init_q_a = np.random.rand()

    estimators = {
        'avg': AverageRewardsEstimator(num_actions),
        'incremental': IncrementalRewardActionValueEstimator(num_actions),
        'constant_step_size': ConstantStepSizeActionValueEstimator(num_actions, 0.1)
    }

    epsilon_generator = epsilon_parameter_generator()
    epsilons = [next(epsilon_generator) for _ in range(10)]
    colors = ['green', 'blue', 'red']
    steps = 200_000
    epochs = 10

    for (estimator_name, estimator), color in zip(estimators.items(), colors):
        total_trailing_avg_rewards = []
        for epsilon in epsilons:
            random_walk_action_reward = RandomWalkActionReward(init_q_a, num_actions)
            action_selector = GreedyActionSelector(estimator, epsilon)
            runner = Runner(random_walk_action_reward, estimator, action_selector)
            trailing_avg_reward = runner.run_epochs(epochs, steps, meter=1, aggregator='trailing_avg', trailing_steps=100_000)
            print("trailing avg: {}".format(trailing_avg_reward))
            total_trailing_avg_rewards.append(trailing_avg_reward)

        plt.plot(epsilons, total_trailing_avg_rewards, color=color, label="estimator=" + estimator_name)

    plt.ylabel("Average reward over last 100,000 steps")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    run()
