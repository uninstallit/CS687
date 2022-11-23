import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pong_gym import Pong
import time

# original source code
# https://keras.io/examples/rl/


def create_model(num_inputs, num_hidden, num_actions):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    common = layers.Dense(num_hidden, activation="relu")(common)
    common = layers.Dense(num_hidden, activation="relu")(common)
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1, activation="relu")(common)
    model = keras.Model(inputs=inputs, outputs=[action, critic])
    return model


def main():

    # Configuration parameters
    # - discount factor for past rewards
    gamma = 0.99
    max_steps_per_episode = 10000
    # - smallest number such that 1.0 + eps != 1.0s
    eps = np.finfo(np.float32).eps.item()

    # Model parameters
    num_inputs = 8
    num_actions = 5
    num_hidden = 128

    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    actor_model = tf.keras.models.load_model("./models/actor")
    l0_weights = actor_model.layers[0].get_weights()
    l1_weights = actor_model.layers[1].get_weights()
    l2_weights = actor_model.layers[2].get_weights()
    l3_weights = actor_model.layers[3].get_weights()
    l4_weights = actor_model.layers[4].get_weights()

    model = create_model(num_inputs, num_hidden, num_actions)
    model.layers[0].set_weights(l0_weights)
    model.layers[1].set_weights(l1_weights)
    model.layers[2].set_weights(l2_weights)
    model.layers[3].set_weights(l3_weights)
    model.layers[4].set_weights(l4_weights)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    huber_loss = keras.losses.Huber()

    # Create the environment
    pong = Pong()
    pong.set_silent(True)

    # run until solved
    while True:
        state = pong.reset()
        episode_reward = 0

        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, axis=0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                # with exploration
                rng = np.random.default_rng()
                action = rng.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # with no exploration
                # action = tf.math.argmax(tf.squeeze(action_probs))
                # action_probs_history.append(tf.math.log(action_probs[0, action]))

                # Apply the sampled action in our environment
                state, reward, done, _ = pong.step(action, timestep)

                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break

            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # log details
        episode_count += 1
        # if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

        if running_reward >= 10:
            pong.set_silent(False)

        # condition to consider the task solved
        if running_reward > 100:
            print("Solved at episode {}!".format(episode_count))
            break


if __name__ == "__main__":
    main()
