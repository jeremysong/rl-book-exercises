"""Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n (the expected number).

To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional 
cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be 
moved from one location to the other in one night. """

import numpy as np


class RentalLocation:
    def __init__(self, request_rate, return_rate, max_cars=20):
        self._request_rate = request_rate
        self._return_rate = return_rate
        self._max_cars = max_cars

    def rent_cars(self, avail_cars):
        """
        Draws number of rental cars from the poisson distribution. If there is no sufficient number of cars, deduct the rewards.
        """
        num_rents = np.random.poisson(self._request_rate)
        if avail_cars >= num_rents:
            reward = num_rents * 10
            left_cars = avail_cars - num_rents
        else:
            reward = avail_cars * 10
            left_cars = 0
        return left_cars, reward

    def return_cars(self, avail_cars):
        num_returns = np.random.poisson(self._request_rate)
        return min(self._max_cars, avail_cars + num_returns)
