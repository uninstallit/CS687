from pong import Pong
from ddpg import DDPG


def main():
    # model parameters
    num_inputs = 8
    num_actions = 5
    lower_bound = 0
    upper_bound = num_actions - 1

    episode = 0
    max_episode = 10000
    max_steps_per_episode = 10000

    # Create the environment
    pong = Pong()
    pong.set_silent(True)

    # left player
    ddpg = DDPG(
        num_inputs,
        lower_bound,
        upper_bound,
        # actor_checkpoint="./checkpoints\ddpg_actor_model.h5",
        # critic_checkpoint="./checkpoints\ddpg_critic_model.h5",
        # target_actor_checkpoint="./checkpoints\ddpg_target_actor.h5",
        # target_critic_checkpoint="./checkpoints\ddpg_target_critic.h5",
    )

    while True:
        prev_state = pong.reset()
        lp_episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            # left player
            lp_action = ddpg.next_action(prev_state)

            # pong arena
            actions = (lp_action, 999)
            state, rewards, done, _ = pong.step(actions, auto_right=True)
            (lp_reward, _) = rewards

            # left player
            lp_episode_reward += lp_reward
            ddpg.learn(prev_state, state, lp_action, lp_reward)

            prev_state = state

            if done:
                break

        episode += 1
        lp_avg_reward = ddpg.update_history(lp_episode_reward)

        # checkpoints
        if episode % 10 == 0:
            # checkpoints
            ddpg.actor_model.save_weights("./checkpoints/ddpg_actor_model.h5")
            ddpg.critic_model.save_weights("./checkpoints/ddpg_critic_model.h5")
            ddpg.target_actor.save_weights("./checkpoints/ddpg_target_actor.h5")
            ddpg.target_critic.save_weights("./checkpoints/ddpg_target_critic.h5")

        print("Episode * {} * Left Avg Reward ==> {}".format(episode, lp_avg_reward))

        if episode == max_episode:
            break


if __name__ == "__main__":
    main()
