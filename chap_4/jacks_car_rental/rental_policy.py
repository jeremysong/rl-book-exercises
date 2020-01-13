import numpy as np


class RentalPolicy:

    def __init__(self):
        self._states_actions = StatesActions(max_cars=20)
        # print("all states {}".format(self._states_actions.get_states()))
        self._state_values = {state: np.random.normal(0, 1) for state in self._states_actions.get_states()}
        # print("state values {}".format(self._state_values))
        # initialize the policy (never moves any cars)
        self._policy = {state: 0 for state, actions in self._states_actions.get_state_actions().items()}
        # self._transition_counter = Counter()  # number of (s', r, s, a)
        # self._state_action_counter = Counter()  # number of (s, a)

    def get_state_values(self):
        return self._state_values

    def get_state_actions(self):
        return self._states_actions.get_state_actions()

    def take_action(self, state):
        action = self._policy[state]  # action is the number of cars to move
        return action

    def update_state_value(self, state, value):
        self._state_values[state] = value

    def update_action(self, state, action):
        self._policy[state] = action

    def get_policy(self):
        return self._policy

    @staticmethod
    def _assert_state(state):
        assert len(state) == 2
        assert 0 <= state[0] <= 20
        assert 0 <= state[1] <= 20


class StatesActions:
    def __init__(self, max_cars=20):
        self._max_cars = max_cars
        self._states_actions = self.generate_states(self._max_cars)

    def generate_states(self, max_cars):
        states = {}
        for x in range(0, max_cars + 1):
            for y in range(0, max_cars + 1):
                states[(x, y)] = self.generate_actions((x, y))

        return states

    def generate_actions(self, avail_car_tuple):
        # Example: (5, 4) -> [-4, -3, -2, ... 4, 5]
        return range(-min(avail_car_tuple[1], 5), min(avail_car_tuple[0], 5) + 1)

    def get_state_actions(self):
        return self._states_actions

    def get_states(self):
        return self._states_actions.keys()


if __name__ == '__main__':
    policy = RentalPolicy()
    print(policy.get_state_values())

    print(policy.get_policy())
