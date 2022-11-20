import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time


def create_actor_model(num_inputs, num_hidden, num_actions):
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(common)
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(common)
    action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    # loss = tf.keras.losses.KLDivergence(reduction="auto", name="kl_divergence")
    # loss = tf.keras.losses.MeanSquaredError(reduction="none")
    loss = tf.keras.losses.MeanAbsoluteError(
        reduction="auto", name="mean_absolute_error"
    )
    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(loss=loss, optimizer=opt)
    return model, loss, opt


def main():
    num_inputs = 8
    num_hidden = 128
    num_actions = 5
    epochs = 25
    history = []

    actor, actor_loss, actor_opt = create_actor_model(
        num_inputs, num_hidden, num_actions
    )

    with open("pong_data.npy", "rb") as f:
        dataset = np.load(f)

    for epoch in range(0, epochs, 1):

        for index, row in enumerate(dataset):
            x = np.expand_dims(row[:num_inputs], axis=0)
            y = int(row[-1])
            labels = np.zeros((num_actions - 1,), dtype=int)
            labels = np.insert(labels, y, 1)
            labels = np.expand_dims(labels, axis=0)

            with tf.GradientTape(persistent=True) as tape:
                action_probs = actor(x)
                actor_loss_value = actor_loss(labels, action_probs)

            actor_gradients = tape.gradient(actor_loss_value, actor.trainable_weights)
            actor_opt.apply_gradients(zip(actor_gradients, actor.trainable_weights))

            if index % 1000 == 0:
                value = actor_loss_value.numpy()
                history.append((value, index))
                print(
                    "epoch: {} - index: {} - loss: {:.3f}".format(epoch, index, value)
                )

    actor.save("./models/actor")

    loss = [p[0] for p in history]
    plt.plot(loss)
    plt.title("Loss History")
    plt.ylabel("loss")
    # plt.xlabel("")
    plt.legend(["train"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
