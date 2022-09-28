import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def create_step_history(step, probabilities, wins, trials, history):
    for bandit_idx in range(0, len(probabilities), 1):
        if trials[bandit_idx] == 0:
            history[bandit_idx][step] = 0.0
        else:
            history[bandit_idx][step] = wins[bandit_idx] / trials[bandit_idx]


# based on source:
# [1] https://peterroelants.github.io/posts/multi-armed-bandit-implementation/
# [3] https://towardsdatascience.com/multi-armed-bandits-thompson-sampling-algorithm-fea205cf31df
def thomson_sampling(probabilities, iterations):
    # init with zeros
    trials = [0.0 for _ in probabilities]
    wins = [0.0 for _ in probabilities]
    history = [{step: 0.0 for step in range(0, iterations)} for _ in probabilities]

    for step in range(0, iterations, 1):
        # conjugate prior - poterior at t + 1 is the prior at t
        bandit_priors = [stats.beta(a=1 + w, b=1 + t - w) for t, w in zip(trials, wins)]
        theta_samples = [dist.rvs(1) for dist in bandit_priors]
        # select the best bandit
        best_bandit = np.argmax(theta_samples)
        # sample best bandit
        rng = np.random.default_rng()
        reward = rng.binomial(1, p=probabilities[best_bandit], size=1)[0]
        # update the observations
        trials[best_bandit] += 1
        wins[best_bandit] += reward
        # record history
        create_step_history(step, probabilities, wins, trials, history)
    return history


def epsilon_greedy(probabilities, iterations, epsilon=0.15):
    # init with zeros
    trials = [0.0 for _ in probabilities]
    wins = [0.0 for _ in probabilities]
    history = [{step: 0.0 for step in range(0, iterations)} for _ in probabilities]

    for step in range(0, iterations, 1):
        rng = np.random.default_rng()
        explore = rng.binomial(1, p=epsilon, size=1)[0]
        # explore
        if explore == 1:
            rng = np.random.default_rng()
            random_bandit = rng.integers(0, high=len(probabilities), size=1)[0]
            # pull random bandit
            rng = np.random.default_rng()
            reward = rng.binomial(1, p=probabilities[random_bandit], size=1)[0]
            trials[random_bandit] += 1
            wins[random_bandit] += reward
        # exploit
        else:
            best_bandit = np.argmax([w / (t + 1) for t, w in zip(trials, wins)])
            rng = np.random.default_rng()
            reward = rng.binomial(1, p=probabilities[best_bandit], size=1)[0]
            # pull bandit with most wins
            rng = np.random.default_rng()
            reward = rng.binomial(1, p=probabilities[best_bandit], size=1)[0]
            trials[best_bandit] += 1
            wins[best_bandit] += reward
            # record history
            create_step_history(step, probabilities, wins, trials, history)
    return history


def upper_confidence_bound(probabilities, iterations):
    # init with zeros
    trials = [0.0 for _ in probabilities]
    wins = [0.0 for _ in probabilities]
    history = [{step: 0.0 for step in range(0, iterations)} for _ in probabilities]

    for step in range(0, iterations, 1):
        # UCB1 policy
        best_bandit = np.argmax(
            [
                w / (t + 1) + np.sqrt(((2 * np.log10(step + 1)) / (t + 1)))
                for t, w in zip(trials, wins)
            ]
        )
        rng = np.random.default_rng()
        reward = rng.binomial(1, p=probabilities[best_bandit], size=1)[0]
        trials[best_bandit] += 1
        wins[best_bandit] += reward
        # record history
        create_step_history(step, probabilities, wins, trials, history)
    return history


def main():
    iterations = 1000
    simulations = 100

    five_armed_probs = [0.3, 0.5, 0.7, 0.83, 0.85]
    eleven_armed_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    probabilities = eleven_armed_probs

    ts_history = []
    eg_history = []
    ucb_history = []
    for _ in range(0, simulations, 1):
        ts_history.append(thomson_sampling(probabilities, iterations))
        eg_history.append(epsilon_greedy(probabilities, iterations))
        ucb_history.append(upper_confidence_bound(probabilities, iterations))

    ts_history = [
        [np.array(list(bandit.values()), dtype=np.dtype("float32")) for bandit in sims]
        for sims in ts_history
    ]
    eg_history = [
        [np.array(list(bandit.values()), dtype=np.dtype("float32")) for bandit in sims]
        for sims in eg_history
    ]
    ucb_history = [
        [np.array(list(bandit.values()), dtype=np.dtype("float32")) for bandit in sims]
        for sims in ucb_history
    ]

    ts_history_mean = np.asarray(ts_history, dtype=object).mean(axis=0)
    eg_history_mean = np.asarray(eg_history, dtype=object).mean(axis=0)
    ucb_history_mean = np.asarray(ucb_history, dtype=object).mean(axis=0)
    history = [ts_history_mean, eg_history_mean, ucb_history_mean]

    fig, axes = plt.subplots(3, 1)
    fig.tight_layout(pad=1.0)
    titles = ["Thomson Sampling", "Epsilon Greedy", "UCB1"]

    for ax, title in zip(axes, titles):
        ax.title.set_text(title)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Reward Probabilities")
        ax.grid(axis="both", color="lightgray")

    arm_steps = [step for step in range(0, iterations, 1)]
    for arm_idx, arm_prob in enumerate(probabilities):
        for algo_idx, _ in enumerate(history):
            arm_means = [history for history in history[algo_idx][arm_idx]]
            axes[algo_idx].step(
                arm_steps,
                arm_means,
                label=f"p={arm_prob}",
            )
            axes[algo_idx].legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()
