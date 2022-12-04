import numpy as np
import tensorflow as tf
from pong import Pong

tf.keras.utils.disable_interactive_logging()


class DeepQ:
    def __init__(
        self, num_states, num_actions, model_checkpoint="", target_checkpoint=""
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        # Configuration paramaters for the whole setup
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        self.batch_size = 32  # Size of batch taken from replay buffer
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        self.model = self.create_q_model()
        if model_checkpoint != "":
            self.model.load_weights(model_checkpoint)
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        self.model_target = self.create_q_model()
        if target_checkpoint != "":
            self.model_target.load_weights(target_checkpoint)
        # In the Deepmind paper they use RMSProp however then Adam optimizer
        # improves training time
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0025, clipnorm=1.0)

        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 50000
        # Number of frames for exploration
        self.epsilon_greedy_frames = 1000000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        self.max_memory_length = 1000000
        # Train the model after 4 actions
        self.update_after_actions = 4
        # How often to update the target network
        self.update_target_network = 10000
        # Using huber loss for stability
        self.loss_function = tf.keras.losses.Huber()

    def create_q_model(self):
        inputs = tf.keras.layers.Input(shape=(self.num_states,))
        common = tf.keras.layers.Dense(128, activation="relu")(inputs)
        common = tf.keras.layers.Dense(128, activation="relu")(common)
        common = tf.keras.layers.Dense(128, activation="relu")(common)
        action = tf.keras.layers.Dense(self.num_actions, activation="linear")(common)
        model = tf.keras.Model(inputs=inputs, outputs=action)
        return model

    def next_action(self, prev_state, frame_count):
        # Use epsilon-greedy for exploration
        if (
            frame_count < self.epsilon_random_frames
            or self.epsilon > np.random.rand(1)[0]
        ):
            # Take random action
            action = np.random.choice(self.num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(prev_state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        # Decay probability of taking random action
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        return action

    def learn(self, state, state_next, action, reward, frame_count, done):
        state_next = np.array(state_next)

        # Save actions and states in replay buffer
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if (
            frame_count % self.update_after_actions == 0
            and len(self.done_history) > self.batch_size
        ):

            # Get indices of samples for replay buffers
            indices = np.random.choice(
                range(len(self.done_history)), size=self.batch_size
            )

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self.model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                future_rewards, axis=1
            )
            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, self.num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if frame_count % self.update_target_network == 0:
            # update the the target network with new weights
            self.model_target.set_weights(self.model.get_weights())

        # Limit the state and reward history
        if len(self.rewards_history) > self.max_memory_length:
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

    def update_history(self, episode_reward):
        # Update running reward to check condition for solving
        self.episode_reward_history.append(episode_reward)
        if len(self.episode_reward_history) > 100:
            del self.episode_reward_history[:1]
        avg_reward = np.mean(self.episode_reward_history)
        return avg_reward


def main():
    # Model parameters
    num_inputs = 8
    num_actions = 5

    episode = 0
    max_steps_per_episode = 10000
    frame_count = 0

    # Create the environment
    pong = Pong()
    pong.set_silent(True)

    deepq = DeepQ(num_inputs, num_actions)

    # Run until solved
    while True:
        prev_state = pong.reset()
        rp_episode_reward = 0

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1

            # right player
            rp_action = deepq.next_action(frame_count, prev_state)

            # Apply the sampled action in our environment
            actions = (999, rp_action)
            state, rewards, done, _ = pong.step(actions, auto_left=True)
            (lp_reward, rp_reward) = rewards

            rp_episode_reward += rp_reward
            deepq.learn(prev_state, state, rp_action, done, frame_count, rp_reward)

        episode += 1
        deepq_avg_reward = deepq.update_history(rp_episode_reward)
        print("Episode * {} * Avg Reward is ==> {}".format(episode, deepq_avg_reward))


if __name__ == "__main__":
    main()
