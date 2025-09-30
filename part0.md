
## Goals

- Understand the key components of a neural network:
  - Layers: Fully connected (Linear), Convolutional, etc.
  - Activation Functions: ReLU, Sigmoid, Tanh
  - Loss Functions: Cross-entropy, MSE
  - Optimizers: SGD, Adam, and their purpose in training
- Learn how forward and backward passes work
- Train basic models using PyTorch on MNIST or CIFAR datasets

---

## Core Concepts

### Layers

Neural networks are composed of layers. The most basic is a fully connected (Linear) layer:

```python
nn.Linear(in_features, out_features)
```

### Activation Functions

Activation functions introduce non-linearity:

```python
nn.ReLU(), nn.Sigmoid(), nn.Tanh()
```

### Loss Functions

Used to measure how far the prediction is from the truth:

```python
nn.CrossEntropyLoss(), nn.MSELoss()
```

### Optimizers

Used to update the model's weights to minimize loss:

```python
torch.optim.SGD(model.parameters(), lr=0.01)
torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## Training Loop Skeleton

```python
for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images).forward()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

