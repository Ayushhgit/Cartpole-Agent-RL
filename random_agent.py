import gymnasium as gym
import cv2
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(1000):

    done = False
    state, info = env.reset()

    while not done:
        frame = env.render()
        cv2.imshow("Cartpole", frame)
        cv2.waitKey(100)

        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        next_step, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
env.close()