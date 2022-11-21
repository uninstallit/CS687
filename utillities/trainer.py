import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def create_actor_model(num_inputs, num_hidden, num_actions):
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(common)
    common = tf.keras.layers.Dense(num_hidden, activation="relu")(common)
    action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
    opt_fn = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model = tf.keras.Model(inputs=inputs, outputs=action)
    model.compile(loss=loss_fn, optimizer=opt_fn)
    return model, loss_fn, opt_fn


def main():
    num_inputs = 8
    num_hidden = 128
    num_actions = 5
    epochs = 200
    batch_size = 128
    history = []

    # create actor model
    actor, actor_loss, actor_opt = create_actor_model(
        num_inputs, num_hidden, num_actions
    )

    # load train data
    num_files = 3
    pong_data = np.empty(shape=(9999, 9))
    for index in range(0, num_files, 1):
        with open(f"./data/pong_data_{index}.npy", "rb") as f:
            data = np.load(f)
            pong_data = np.row_stack([pong_data, data])

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    for epoch in range(0, epochs, 1):
        tmp_hist = []

        for index, batch in enumerate(dataset):
            x = batch[:, :num_inputs]
            y = tf.keras.utils.to_categorical(batch[:, -1], num_classes=num_actions)
            with tf.GradientTape(persistent=True) as tape:
                action_probs = actor(x)
                actor_loss_value = actor_loss(y, action_probs)

            actor_gradients = tape.gradient(actor_loss_value, actor.trainable_weights)
            actor_opt.apply_gradients(zip(actor_gradients, actor.trainable_weights))

            loss_value = actor_loss_value.numpy()
            tmp_hist.append(loss_value)

            if index % 1000 == 0:
                print(
                    "epoch: {} - index: {} - loss: {:.3f}".format(
                        epoch, index, loss_value
                    )
                )

        average_loss = sum(tmp_hist) / len(tmp_hist)
        history.append(average_loss)

    # save model
    actor.save("./models/actor")

    # plot data
    plt.plot(history)
    plt.title("Loss History")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
