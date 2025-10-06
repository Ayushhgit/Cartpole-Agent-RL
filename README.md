# SARSA & Q-Learning with Neural Networks on CartPole

##### CartPole Results

![vid_0_14](https://user-images.githubusercontent.com/53657825/178179457-d88f9844-c20f-4344-84e5-ccd3436cb4cb.gif)

This repository contains implementations of **SARSA** and **Q-Learning** algorithms to solve the **CartPole-v1** environment from [OpenAI Gym](https://www.gymlibrary.dev/). Unlike the classical tabular methods, this project uses **Neural Networks** as function approximators for the Q-value function.

---

## 📌 Problem Statement

The **CartPole** environment is a reinforcement learning benchmark where the agent must balance a pole on a moving cart.

* **State space**: Continuous (position, velocity, angle, angular velocity)
* **Action space**: Discrete (move left or right)
* **Goal**: Maximize the time steps before the pole falls or cart goes out of bounds.

---

## 🧠 Algorithms Implemented

### 1. Q-Learning

* **Off-policy** Temporal-Difference control algorithm.
* Uses the maximum estimated Q-value for the next state to update the current state-action pair.
* Update rule:

[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big]
]

### 2. SARSA (State-Action-Reward-State-Action)

* **On-policy** TD control algorithm.
* Uses the action actually taken in the next state (following the current policy) for updates.
* Update rule:

[
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma Q(s',a') - Q(s,a)\Big]
]

---

## 🧮 Function Approximation with Neural Networks

Since the CartPole environment has a **continuous state space**, tabular methods are infeasible.

* A **Neural Network (NN)** is used to approximate the Q-function:

[
Q(s, a; \theta) \approx \text{NN}(s)
]

* Architecture:

  * Input: 4-dimensional state vector
  * Hidden layers: Dense layers with ReLU activation
  * Output: 2 values (Q-value for each action)

* The network is trained using gradient descent to minimize the TD-error:

[
L = \Big[ r + \gamma Q(s', a') - Q(s, a) \Big]^2
]

---

## ⚙️ Requirements

* Python 3.9+
* TensorFlow / PyTorch (depending on implementation)
* Gymnasium (for CartPole-v1)
* Numpy

## 📊 Results

* Both SARSA and Q-Learning agents learn to balance the pole.
* Q-Learning typically converges faster (off-policy), while SARSA is more conservative but stable.
* With NN approximation, the agents generalize across continuous states instead of storing explicit Q-tables.


## 📖 References

* OpenAI Gym Documentation: [https://www.gymlibrary.dev/](https://www.gymlibrary.dev/)
* [CartPole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

---

## ✨ Future Work

* Implement **Deep Q-Network (DQN)** with target networks & experience replay.
* Compare performance with **Policy Gradient** methods.
* Hyperparameter tuning for better convergence.


