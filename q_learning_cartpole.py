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
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 100

def policy(state, explore=0.0):
    if tf.random.uniform(()) < explore:
        return tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    q_values = q_net(state, training=False)
    return tf.argmax(q_values[0], axis=-1)

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    state = tf.convert_to_tensor([obs], dtype=tf.float32)

    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        # ε-greedy action
        action = policy(state, EPSILON)

        # Step
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated

        next_state = tf.convert_to_tensor([next_obs], dtype=tf.float32)

        # Q-learning target (max over actions)
        target = reward
        if not done:
            next_q = q_net(next_state, training=False)
            target += GAMMA * tf.reduce_max(next_q[0])

        # Update Q-network
        with tf.GradientTape() as tape:
            q_values = q_net(state)
            current_q = q_values[0][action]
            loss = tf.square(target - current_q)

        grads = tape.gradient(loss, q_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

        # Transition
        state = next_state
        total_reward += reward
        episode_length += 1

    print(f"Episode {episode}, Length {episode_length}, Reward {total_reward}, Epsilon {EPSILON:.4f}")
    EPSILON = max(0.01, EPSILON / EPSILON_DECAY)

q_net.save("q_learning_q_net.h5")
env.close()
