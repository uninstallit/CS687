import time
import numpy as np
from gym import Pong


def main():

    pong = Pong()
    state = pong.reset()
    pong.set_silent(False)
    max_steps_per_episode = 10000

    data = []

    for timestep in range(1, max_steps_per_episode):

        # actuon 5: do nothing default
        action = 5
        state, reward, done, label = pong.step(action, timestep)

        time.sleep(0.05)

        row = state + [label]
        data.append(row)

        if timestep % 100 == 0:
            print("timestep - ", timestep)

    with open("pong_data.npy", "wb") as f:
        np.save(f, data)


if __name__ == "__main__":
    main()
