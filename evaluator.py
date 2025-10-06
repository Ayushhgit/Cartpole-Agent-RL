import gymnasium as gym
import cv2
import tensorflow as tf
from keras.models import load_model

# Environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
q_net = load_model("sarsa_q_net.h5")

def policy(state, explore=0.0):
    q_values = q_net(state, training=False)
    action = tf.argmax(q_values[0], axis=-1)
    if tf.random.uniform(()) < explore:
        action = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)
    return action

for episode in range(20):
    obs, info = env.reset()
    state = tf.convert_to_tensor([obs], dtype=tf.float32)
    done = False

    while not done:
        # Render
        frame = env.render()
        cv2.imshow("CartPole", frame)
        cv2.waitKey(50)   # Lower for faster playback

        # Action selection
        action = policy(state)

        # Environment step
        obs, reward, terminated, truncated, _ = env.step(action.numpy())
        state = tf.convert_to_tensor([obs], dtype=tf.float32)
        done = terminated or truncated
    print(f'Episode:{episode+1}')


env.close()
cv2.destroyAllWindows()
