import random
import numpy as np
import matplotlib.pyplot as plt


def value_iteration(
    p, states, actions, rewards, policy, values, theta=0.005, gamma=0.9
):
    while True:
        delta = 0

        for state in states:
            # immediate reward
            immediate_state_reward = rewards[state]
            temp_value = values[state]
            expected_value = []

            if state in policy:

                for action in actions[state]:
                    win_pot = 2 * action
                    diff = state % action

                    next_state_win_flip = min([win_pot + diff, 100])
                    next_state_lose_flip = state - action

                    ex_reward = (p * rewards[next_state_win_flip]) + (
                        (1 - p) * rewards[next_state_lose_flip]
                    )
                    expected_value.append((ex_reward, action))

                # (max_ex, max_action) = sorted(
                #     expected_value, key=lambda x: x[0], reverse=True
                # )[0]

                # minmax - tie-breaking policy
                items = sorted(expected_value, key=lambda x: x[0], reverse=True)
                high_score = items[0][0] # (reward, action)
                items = [item for item in items if item[0] == high_score]
                (max_ex, max_action) = sorted(items, key=lambda x: x[1])[0]
                
                values[state] = immediate_state_reward + gamma * max_ex
                policy[state] = max_action

                delta = max([delta, abs(temp_value - values[state])])

        if delta < theta:
            break

    return states, values, policy


def main():

    max_amount = 100

    # If the coin comes up heads you double the amount you bet,
    # and if it comes up tails you lose the entire amount you bet
    p_coin = [0.25, 0.4, 0.55]

    # states: The state is your capital dollar = {0, 1, 2, ..., 99, 100 }
    # terminal states:
    # - accumulate $100 (you win),
    # - or go bust (end up with $0 and lose).
    states = tuple(np.flip(np.arange(0, max_amount + 1, 1, dtype=np.int16), axis=0))

    # rewards: 100 is +1, and 0 otherwise
    rewards = {}
    for state in states:
        if state == 100:
            rewards[state] = 1
        else:
            rewards[state] = 0

    # actions: The actions are the amounts you may choose to bet.
    actions = {}
    for state in states:
        if state != 0 and state != 100:
            bets = np.flip(np.arange(0, state + 1, 1, dtype=np.int16), axis=0)
            actions[state] = tuple(bets)
        else:
            actions[state] = tuple([0])

    # initial policy
    policy = {}
    for state in actions.keys():
        policy[state] = random.choice(actions[state])

    # initialize value - how good is a state for the agent to be in
    values = {}
    for state in states:
        if state in actions.keys():
            values[state] = 0
        # terminal
        if state == 0:
            values[state] = 0
        # terminal
        if state == 100:
            values[state] = 1

    # plots

    states_history = []
    values_history = []
    policy_history = []

    for p in p_coin:
        states, values, policy = value_iteration(
            p, states, actions, rewards, policy, values, gamma=0.9
        )
        states_history.append(np.asarray(states))
        values_history.append([values[state] for state in states])
        policy_history.append([policy[state] for state in states])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Value Iteration Algorithm")
    ax1.set_xlabel("state")
    ax1.set_ylabel("value")
    ax1.grid(axis="x", color="0.95")
    ax2.set_xlabel("state")
    ax2.set_ylabel("policy")
    ax2.grid(axis="x", color="0.95")

    for index, p in enumerate(p_coin):
        ax1.step(states_history[index], values_history[index], label=f"p={p}")
        ax1.legend(loc="upper left")
        ax2.step(states_history[index], policy_history[index], label=f"p={p}")
        ax2.legend(loc="upper left")

    plt.show()





if __name__ == "__main__":
    main()
