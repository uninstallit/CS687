from pong import Pong
from deepq import DeepQ


def main():
    # model parameters
    num_inputs = 8
    num_actions = 5

    episode = 0
    max_episode = 1000
    max_steps_per_episode = 10000
    frame_count = 0

    # Create the environment
    pong = Pong()
    pong.set_silent(False)

    # right player
    deepq = DeepQ(num_inputs, num_actions)

    while True:
        prev_state = pong.reset()
        rp_episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # right player
            rp_action = deepq.next_action(prev_state, frame_count)

            # pong arena
            actions = (999, rp_action)
            state, rewards, done, _ = pong.step(actions, auto_left=True)
            (_, rp_reward) = rewards

            # right player
            rp_episode_reward += rp_reward
            deepq.learn(prev_state, state, rp_action, rp_reward, frame_count, done)

            prev_state = state
            frame_count = frame_count + 1

            if done:
                break

        episode += 1
        rp_avg_reward = deepq.update_history(rp_episode_reward)

        # checkpointsh  
        if episode % 10 == 0:
            # checkpoints
            deepq.model.save_weights("./checkpoints/deepq_model.h5")
            deepq.model_target.save_weights("./checkpoints/deepq_model_target.h5")

        print("Episode * {} * Right Avg Reward ==> {}".format(episode, rp_avg_reward))

        if episode == max_episode:
            break


if __name__ == "__main__":
    main()
