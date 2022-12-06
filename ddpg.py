import numpy as np
import tensorflow as tf
from pong import Pong


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(
        self,
        num_states,
        target_actor,
        target_critic,
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        gamma,
        buffer_capacity=1000000,
        batch_size=32,
    ):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        self.num_states = num_states
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.actor_model = actor_model
        self.critic_model = critic_model

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, 1))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )

            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        self.update(state_batch, action_batch, reward_batch, next_state_batch)


class DDPG:
    def __init__(
        self,
        num_states,
        lower_bound,
        upper_bound,
        actor_checkpoint="",
        critic_checkpoint="",
        target_actor_checkpoint="",
        target_critic_checkpoint="",
    ):
        self.num_states = num_states
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # hyperparams
        self.std_dev = 0.5
        self.ou_noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1)
        )

        self.actor_model = self.get_actor()
        if actor_checkpoint != "":
            self.actor_model.load_weights(actor_checkpoint)

        self.critic_model = self.get_critic()
        if critic_checkpoint != "":
            self.critic_model.load_weights(critic_checkpoint)

        self.target_actor = self.get_actor()
        if target_actor_checkpoint != "":
            self.target_actor.load_weights(target_actor_checkpoint)

        self.target_critic = self.get_critic()
        if target_critic_checkpoint != "":
            self.target_critic.load_weights(target_critic_checkpoint)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr, clipnorm=1.0)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr, clipnorm=1.0)

        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.001

        self.buffer = Buffer(
            num_states=self.num_states,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            actor_model=self.actor_model,
            critic_model=self.critic_model,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            gamma=self.gamma,
            buffer_capacity=1000000,
            batch_size=32,
        )

        # To store reward history of each episode
        self.ep_reward_list = []
        # To store average reward history of last few episodes
        self.avg_reward_list = []

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = tf.keras.layers.Input(shape=(self.num_states,))
        out = tf.keras.layers.Dense(128, activation="relu")(inputs)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        outputs = tf.keras.layers.Dense(
            1, activation="tanh", kernel_initializer=last_init
        )(out)
        outputs = tf.math.abs(outputs * self.upper_bound)
        # outputs = outputs * (self.upper_bound / 2.0) + (self.upper_bound / 2.0)
        actor = tf.keras.Model(inputs, outputs)
        return actor

    def get_critic(self):
        # State as input
        state_input = tf.keras.layers.Input(shape=(self.num_states,))
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_input)
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_out)
        # Action as input
        action_input = tf.keras.layers.Input(shape=(1,))
        action_out = tf.keras.layers.Dense(128, activation="relu")(action_input)
        # Both are passed through seperate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])
        out = tf.keras.layers.Dense(128, activation="relu")(concat)
        out = tf.keras.layers.Dense(128, activation="relu")(out)
        outputs = tf.keras.layers.Dense(1)(out)
        # Outputs single value for give state-action
        critic = tf.keras.Model([state_input, action_input], outputs)
        return critic

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return np.squeeze(legal_action)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        result = []
        for (a, b) in zip(target_weights, weights):
            a = b * tau + a * (1 - tau)
            result.append(a)
        return result

    def next_action(self, prev_state):
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = self.policy(state=tf_prev_state)
        return action

    def learn(self, prev_state, state, action, reward):
        self.buffer.record((prev_state, action, reward, state))
        self.buffer.learn()
        result = self.update_target(
            self.target_actor.variables, self.actor_model.variables, self.tau
        )
        self.target_actor.set_weights(result)
        
        result = self.update_target(
            self.target_critic.variables, self.critic_model.variables, self.tau
        )
        self.target_critic.set_weights(result)

    def update_history(self, episodic_reward):
        self.ep_reward_list.append(episodic_reward)
        # Mean of last 40 episodes
        avg_reward = np.mean(self.ep_reward_list[-40:])
        self.avg_reward_list.append(avg_reward)
        return avg_reward
