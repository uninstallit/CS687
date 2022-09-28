import numpy as np
import matplotlib.pyplot as plt


def flip(p):
    rng = np.random.default_rng()
    flip = rng.binomial(1, p=p, size=1)[0]
    return flip


def main():

    p = 0.5
    dollars = 0

    iterations = []
    for n in range(1, 9, 1):
        iterations.append(10 ** n)

    history = []

    for iteration in iterations:
        payoffs = []

        for _ in range(0, iteration, 1):
            k = 1

            while flip(p) == 0:
                k = k + 1

            payoffs.append(2 ** k)

    history.append(np.asarray(payoffs).mean())

    fig, ax = plt.subplots(3, 1)
    ax.plot(iterations, history)
    plt.show()


if __name__ == "__main__":
    main()
