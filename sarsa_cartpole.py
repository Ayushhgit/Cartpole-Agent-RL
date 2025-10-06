import gymnasium as gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

# Environment
env = gym.make("CartPole-v1")

# Q-Network
net_input = Input(shape=(4,))
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Parameters
EPSILON = 0.1
GAMMA = 0.99
NUM_EPISODES = 50


def policy(state, explore=0.0):
    if tf.random.uniform(shape=()) < explore:
        return tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    q_values = q_net(state)
    return tf.argmax(q_values[0])


for episode in range(NUM_EPISODES):

    obs, info = env.reset()
    state = tf.convert_to_tensor([obs], dtype=tf.float32)
    action = policy(state, EPSILON)

    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated

        next_state = tf.convert_to_tensor([next_obs], dtype=tf.float32)
        next_action = policy(next_state, EPSILON)

        target = reward
        if not done:
            target += GAMMA * q_net(next_state)[0][next_action]

        with tf.GradientTape() as tape:
            q_values = q_net(state)
            current_q = q_values[0][action]
            loss = tf.reduce_mean(tf.square(target - current_q))


        grads = tape.gradient(loss, q_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1

    print(f"Episode {episode}, Length {episode_length}, Reward {total_reward}, Epsilon {EPSILON:.4f}")
    EPSILON = max(0.01, EPSILON * 0.995)


q_net.save("sarsa_q_net.h5")
env.close()
