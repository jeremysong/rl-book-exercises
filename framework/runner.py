import numpy as np


class Runner:
    def __init__(self, action_reward, action_value_estimator, action_selector):
        self._action_reward = action_reward
        self._action_value_estimator = action_value_estimator
        self._action_selector = action_selector

    def run_steps(self, steps=1000):
        """
        Runs simulation with number of steps.
        :return: the average reward for each step and optimal action for each step
        """
        # reset to have a fresh start in each epoch
        self._action_value_estimator.reset()
        self._action_reward.reset()

        avg_rewards = []
        # indicate if an optimal action is made for each step
        optimal_actions = []

        for step in range(0, steps):
            action = self._action_selector.select_action()
            if self._action_reward.get_optimal_action() is not None:
                if action == self._action_reward.get_optimal_action():
                    optimal_actions.append(1)
                else:
                    optimal_actions.append(0)
            reward = self._action_reward.get_reward(action)
            # print("Selected action: {} with record: {}".format(action, reward))
            self._action_value_estimator.add_reward(action, reward)
            avg_rewards.append(self._action_value_estimator.get_avg_reward())

        return avg_rewards, optimal_actions

    def run_epochs(self, epochs=2000, steps=1000, meter=200, **kwargs):
        """
        Runs simulation with number of epochs. :meter: decides the frequency of printing out epoch number 
        :aggregator: 'avg_per_step' returns the average reward per epoch for every step, the optimal action 
        percentage per epoch for every step. 'trailing_steps' returns the average rewards of the trailing N stpes, specificed by 'trailing_steps' 
        """
        total_avg_rewards = [0.0] * steps
        total_optimal_actions = [0] * steps

        for epoch in range(0, epochs):
            if epoch > 0 and epoch % meter == 0:
                print("Running epoch: {}".format(epoch))
            avg_rewards, optimal_actions = self.run_steps(steps)
            total_avg_rewards = [x + y for x, y in zip(total_avg_rewards, avg_rewards)]
            total_optimal_actions = [x + y for x, y in zip(total_optimal_actions, optimal_actions)]

        aggregator_type = kwargs.get('aggregator')
        if aggregator_type is None or aggregator_type == 'avg_per_step':
            return np.array(total_avg_rewards) / epochs, np.array(total_optimal_actions) / epochs
        elif aggregator_type == 'trailing_avg':
            trailing_steps = kwargs.get('trailing_steps')
            return sum(total_avg_rewards[-trailing_steps:]) / trailing_steps / epochs
