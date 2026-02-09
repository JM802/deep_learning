# Linear Neural Network

 **Focus** : Data Flow, Dimensionality, and SGD Implementation Logic

## 1. Architectural Overview

A linear neural network maps an input **$\mathbf{X} \in \mathbb{R}^{n \times d}$** to a prediction **$\hat{\mathbf{y}} \in \mathbb{R}^{n \times 1}$** via a weight matrix **$\mathbf{w}$** and a bias **$b$**.

### Core Components:

1. **Data Pipeline** : Shuffling(打乱) and Mini-batching.
2. **Model Archetype(原型)** : **$y = \mathbf{X}\mathbf{w} + b$**.
3. **Loss Logic** : Squared Error with mean reduction.
4. **Optimization** : Stochastic Gradient Descent (SGD).

---

## 2. Mathematical Consistency(一致性) & Dimensionality

To avoid `Broadcasting Errors`, we must strictly enforce tensor shapes:

---

## 3. Atomic Logic Implementation

### A. Data Iterator (Shuffling & Batching)

Instead of moving heavy data, we manipulate(操作) indices.

### B. The Optimization Step (SGD)

 **Logic** : We update parameters in the opposite direction of the gradient.

> **Critical Design** : Since we use `.mean()` in the loss, the gradient is already averaged. Do not divide by `batch_size` again in the optimizer.

---

## 4. The Training Loop (Decision Tree)

1. **Initialize** : Generate **$\mathbf{w} \sim \mathcal{N}(0, 0.01)$** and **$b = 0$**.
2. **Forward Pass** : Compute **$\hat{y} = \mathbf{X}\mathbf{w} + b$**.
3. **Loss Calculation** :

$$
$$L = \frac{1}{2n} \sum (\hat{y} - y)^2
$$

   *Note: `reshape` is used to ensure **$y$** and **$\hat{y}$** are both **$(n, 1)$**.*

1. **Backward Pass** : `loss.mean().backward()` to compute **$\nabla_{\mathbf{w}}L$** and **$\nabla_{b}L$**.
2. **Parameter Update** : Invoke `sgd`.

---

## 5. Engineering Quality Control (Unit Tests)

To ensure the system's robustness, always assert your assumptions:
