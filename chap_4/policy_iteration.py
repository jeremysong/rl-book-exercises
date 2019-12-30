from scipy.stats import poisson


class PolicyIteration:
    def __init__(self, policy, gamma=0.9):
        self._policy = policy
        self._gamma = gamma
        self._car_rental = {
            'rental': [poisson(3), poisson(3)],
            'return': [poisson(4), poisson(2)]
        }

    def evaluate(self):
        """
        Evaluates the policy, then improve it by updating the policy with best action
        """
        while True:
            delta = 0
            for state, value in self._policy.get_state_values().items():
                action = self._policy.take_action(state)
                new_value = self._estimate_value(state, action)
                self._policy.update_state_value(state, new_value)
                delta = max(delta, abs(value - new_value))
                print("Updating value for state {} from {} to {} with delta {}".format(state, value, new_value, delta))

            if delta < 0.005:
                break

        print("policy state values: {}".format(self._policy.get_state_values()))
        self.improvement()

    def improvement(self):
        """
        Improves the policy by finding and update the best action for each state. If the policy is not stable,
        evaluate it again.
        """
        policy_stable = True
        for state, value in self._policy.get_state_values().items():
            old_action = self._policy.take_action(state)
            best_action = self._find_best_action(state)
            self._policy.update_action(state, best_action)
            if old_action != best_action:
                policy_stable = False

        print("new policy: {}".format(self._policy.get_policy()))

        if policy_stable:
            return
        else:
            self.evaluate()

    def _estimate_value(self, state, action):
        total_reward = 0
        # number of cars start of the day
        new_state, reward = self._move_cars(state, action)
        total_reward += reward

        for num_rent_1 in range(0, new_state[0] + 1):
            for num_rent_2 in range(0, new_state[1] + 1):
                reward = (num_rent_1 + num_rent_2) * 10
                for num_return_1 in range(0, 11):  # only consider returning less than 11 cars to reduce the computation load
                    for num_return_2 in range(0, 11):
                        # Number of cars by end of day
                        car_1 = min(new_state[0] - num_rent_1 + num_return_1, 20)
                        car_2 = min(new_state[1] - num_rent_2 + num_return_2, 20)
                        prob = self._car_rental['rental'][0].pmf(num_rent_1) * \
                               self._car_rental['rental'][1].pmf(num_rent_2) * \
                               self._car_rental['return'][0].pmf(car_1 - (new_state[0] - num_rent_1)) * \
                               self._car_rental['return'][1].pmf(car_2 - (new_state[1] - num_rent_2))

                        total_reward += prob * (reward + self._gamma * self._policy.get_state_values()[(car_1, car_2)])

        return total_reward

    @staticmethod
    def _move_cars(state, action):
        new_state = (state[0] - action, state[1] + action)
        reward = abs(action) * (-2)
        return new_state, reward

    def _find_best_action(self, state):
        action_values = {action: self._estimate_value(state, action) for action in
                         self._policy.get_state_actions()[state]}
        # print("estimated action value: {}".format(list(action_values)))
        best_action = max(action_values, key=action_values.get)
        # print("best action {} with value {} for state {}".format(best_action, action_values[best_action], state))

        return best_action
