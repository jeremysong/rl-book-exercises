from chap_4.policy_iteration import PolicyIteration
from chap_4.rental_policy import RentalPolicy

if __name__ == '__main__':
    rental_policy = RentalPolicy()
    print(rental_policy.get_state_values())
    print(rental_policy.get_policy())
    policy_iteration = PolicyIteration(rental_policy)

    policy_iteration.evaluate()
    print(rental_policy.get_state_values())
    print("Optimal policy {}".format(rental_policy.get_policy()))
