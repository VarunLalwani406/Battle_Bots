

## Goals

* Understand why tabular Q-learning fails in large state spaces
* Learn how DQNs approximate the Q-function using neural networks
* Implement a simple DQN in PyTorch
* Train a DQN on OpenAI Gymnasium's CartPole environment
* Extend your DQN agent to train on your own Snake game

---

## Core Concepts

### Function Approximation with Neural Networks

In DQNs, we replace the Q-table with a neural network:

$$
Q(s, a; \theta) \approx \text{expected return}
$$

Where $\theta$ are the parameters of the neural network.

### Experience Replay

Instead of learning from recent transitions only, we store experiences in a replay buffer and sample batches from it:

* Breaks correlation between subsequent steps
* Improves sample efficiency

### Target Network

We use a separate, periodically-updated target network $Q_{target}(s, a; \theta^-)$ to stabilize learning:

$$
\text{loss} = \left(Q(s, a) - \left(\text{reward} + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a')\right)\right)^2
$$

### DQN Training Loop Skeleton

```python
for episode in range(num_episodes):
    state, _ = env.reset()

    for t in range(max_timesteps):
        # ε-greedy policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_net(state))

        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))

        # Sample batch and compute loss
        if len(replay_buffer) > batch_size:
            batch = sample(replay_buffer)
            optimize(q_net, target_net, batch)

        if done:
            break
        state = next_state
```

---




