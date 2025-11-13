## Research Notes: Modules 4 & 5 - Artificial Neural Networks (Maths)

**Objective:** To understand Artificial Neural Networks (ANNs), their mathematical formulation, and the backpropagation algorithm, with a rigorous focus on the dimensions of the tensors involved.

---

### 1. From Linear Models to Neurons

A linear model (like Linear Regression) is a single computational unit. For an input vector $x$ and parameters $W$ and $b$, the prediction $\hat{y}$ is:
$$
\hat{y} = W^T x + b
$$
* **Dimensions (Single Input, Single Output):**
    * Input features: $x \in \mathbb{R}^{d_{in}}$ (a $d_{in} \times 1$ column vector).
    * Weights: $W \in \mathbb{R}^{d_{in}}$ (a $d_{in} \times 1$ column vector).
    * Bias: $b \in \mathbb{R}$ (a scalar).
    * Output: $\hat{y} \in \mathbb{R}$ (a scalar).

An **artificial neuron** adds a non-linear **activation function** $f(\cdot)$ to this linear combination.

* **Neuron Computation:**
    1.  **Linear Combination (Logit):** $z = W^T x + b \quad \to \quad z \in \mathbb{R}$
    2.  **Activation:** $\hat{y} = f(z) \quad \to \quad \hat{y} \in \mathbb{R}$

* **Common Activation Functions $f(z)$ & Their Derivatives:**
    * **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$.
        * Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
    * **Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$.
        * Derivative: $\tanh'(z) = 1 - \tanh^2(z)$.
    * **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$.
        * Derivative: $f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$.

---

### 2. Multi-Layer Perceptron (MLP)

An MLP stacks layers of neurons. The output of one layer becomes the input to the next.

* **Architecture (2-Layer Example):**
    * Input Layer ($d_{in}$ features)
    * Hidden Layer ($d_h$ neurons)
    * Output Layer ($d_{out}$ neurons)

#### 2.1. Forward Pass: Batch (Tensor) Perspective
This is the standard, efficient implementation using **Tensors** (multi-dimensional arrays). We use the $Z = XW$ convention.

* **Input Tensor $X$:** $X \in \mathbb{R}^{N \times d_{in}}$ ($N$ = batch size).

* **1. Hidden Layer $H$:**
    * Weights $W_h \in \mathbb{R}^{d_{in} \times d_h}$.
    * Bias $b_h \in \mathbb{R}^{d_h}$ (or $1 \times d_h$, which is broadcast).
    * Logits $Z_h = X W_h + b_h$
        * *Dimensions: $(N \times d_{in}) \cdot (d_{in} \times d_h) + (1 \times d_h) \to (N \times d_h)$*
    * Activations $H = f_h(Z_h)$ (element-wise).
        * *Dimensions: $H \in \mathbb{R}^{N \times d_h}$*

* **2. Output Layer $\hat{Y}$:**
    * Weights $W_y \in \mathbb{R}^{d_h \times d_{out}}$.
    * Bias $b_y \in \mathbb{R}^{d_{out}}$ (or $1 \times d_{out}$, broadcast).
    * Logits $Z_y = H W_y + b_y$
        * *Dimensions: $(N \times d_h) \cdot (d_h \times d_{out}) + (1 \times d_{out}) \to (N \times d_{out})$*
    * Prediction $\hat{Y} = f_y(Z_y)$ (element-wise).
        * *Dimensions: $\hat{Y} \in \mathbb{R}^{N \times d_{out}}$*

---

### 3. Loss Functions (Mathematical Derivation)

**Goal:** Find parameters $\theta = \{W_h, b_h, W_y, b_y\}$ that minimize a **Loss Function** $J(\theta)$.

* **MSE (Regression):** The loss is the Mean Squared Error.
    $$
    J(\theta) = \frac{1}{N} \sum_{i=1}^N \frac{1}{2} ||y_i - \hat{y}_i||_2^2 = \frac{1}{2N} ||Y - \hat{Y}||_F^2
    $$
    * The gradient of the loss w.r.t. the final prediction (used for backprop) is:
        $$
        \frac{\partial J}{\partial \hat{Y}} = \frac{1}{N}(\hat{Y} - Y)
        $$

* **Cross-Entropy (Classification):** Derived from Maximum Likelihood Estimation (MLE).
    * **Likelihood:** We want to find $\theta$ that maximizes the probability of observing our data.
        $$
        \mathcal{L}(\theta) = \prod_{i=1}^N p(y_i \mid x_i; \theta)
        $$
    * **Log-Likelihood:** Maximizing the log is equivalent and computationally easier.
        $$
        \log \mathcal{L}(\theta) = \sum_{i=1}^N \log p(y_i \mid x_i; \theta)
        $$
    * **Loss:** The NLL is the loss $J(\theta) = - \log \mathcal{L}(\theta)$. Minimizing NLL is maximizing likelihood.
    
    * **Binary Cross-Entropy (BCE):** For $y \in \{0, 1\}$, $\hat{y} = \sigma(z_y)$. The likelihood is $p(y|x) = \hat{y}^y (1-\hat{y})^{1-y}$.
        * **NLL Loss:** $J(\theta) = - \sum_{i=1}^N \left( y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i) \right)$.
    
    * **Categorical Cross-Entropy (CCE):** For $K$ classes, $y$ is one-hot, $\hat{y} = \text{softmax}(z_y)$. Likelihood $p(y|x) = \prod_{k=1}^K \hat{y}_k^{y_k}$.
        * **NLL Loss:** $J(\theta) = - \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{y}_{ik}$.

---

### 4. The Backpropagation Algorithm (Dimensional Analysis)

Backpropagation is an algorithm to compute the gradient $\nabla_\theta J$ by recursively applying the chain rule. We derive the gradients for the **batched (Tensor) implementation** ($Z=XW$).

**Notation:**
* $J$ is the scalar loss.
* $\frac{\partial J}{\partial W_y}$ is the gradient (a matrix, same shape as $W_y$).
* $\frac{\partial J}{\partial Z_y}$ is the gradient of the loss w.r.t. the output logits (a tensor, same shape as $Z_y$).
* $\odot$ is the Hadamard (element-wise) product.

**1. Gradient w.r.t. Output Layer ($\nabla_{W_y} J$, $\nabla_{b_y} J$)**

* **a) Gradient at Output Logits $\frac{\partial J}{\partial Z_y}$:**
    This is the "error" signal. We apply the chain rule:
    $$
    \frac{\partial J}{\partial Z_y} = \frac{\partial J}{\partial \hat{Y}} \odot \frac{\partial \hat{Y}}{\partial Z_y} = \frac{\partial J}{\partial \hat{Y}} \odot f_y'(Z_y)
    $$
    * *Dimensions: $(N \times d_{out}) \odot (N \times d_{out}) \to (N \times d_{out})$*
    * **Special Case (Softmax + CCE):** This derivative simplifies beautifully:
        $$
        \frac{\partial J}{\partial Z_y} = \frac{1}{N}(\hat{Y} - Y)
        $$

* **b) Gradient for Weights $\nabla_{W_y} J$:**
    Using the chain rule for matrix calculus: $\frac{\partial J}{\partial W} = X^T \frac{\partial J}{\partial Z}$. Here, the "input" to the layer was $H$.
    $$
    \nabla_{W_y} J = H^T \frac{\partial J}{\partial Z_y}
    $$
    * *Dimensions: $(d_h \times N) \cdot (N \times d_{out}) \to (d_h \times d_{out})$.* (Matches $W_y$ shape)

* **c) Gradient for Bias $\nabla_{b_y} J$:**
    The bias $b_y$ was broadcast (added) to every sample. Its gradient is the sum of the incoming gradients across the batch dimension.
    $$
    \nabla_{b_y} J = \sum_{i=1}^N \left(\frac{\partial J}{\partial Z_y}\right)_i \quad (\text{Sum over batch axis } 0)
    $$
    * *Dimensions: Summing $(N \times d_{out})$ over axis 0 $\to (d_{out})$.* (Matches $b_y$ shape)

**2. Gradient w.r.t. Hidden Layer ($\nabla_{W_h} J$, $\nabla_{b_h} J$)**

* **a) Backpropagate Error to $H$:**
    First, propagate the error from $Z_y$ back to the hidden layer's output $H$, using the chain rule: $\frac{\partial J}{\partial X} = \frac{\partial J}{\partial Z} W^T$.
    $$
    \frac{\partial J}{\partial H} = \frac{\partial J}{\partial Z_y} W_y^T
    $$
    * *Dimensions: $(N \times d_{out}) \cdot (d_{out} \times d_h) \to (N \times d_h)$.* (Matches $H$ shape)

* **b) Gradient at Hidden Logits $\frac{\partial J}{\partial Z_h}$:**
    Now, pass this error back through the hidden layer's activation $f_h$.
    $$
    \frac{\partial J}{\partial Z_h} = \frac{\partial J}{\partial H} \odot f_h'(Z_h)
    $$
    * *Dimensions: $(N \times d_h) \odot (N \times d_h) \to (N \times d_h)$.* (Matches $Z_h$ shape)

* **c) Gradient for Weights $\nabla_{W_h} J$:**
    Now we have the gradient w.r.t. $Z_h$. We can find the gradient for $W_h$. The "input" to this layer was $X$.
    $$
    \nabla_{W_h} J = X^T \frac{\partial J}{\partial Z_h}
    $$
    * *Dimensions: $(d_{in} \times N) \cdot (N \times d_h) \to (d_{in} \times d_h)$.* (Matches $W_h$ shape)

* **d) Gradient for Bias $\nabla_{b_h} J$:**
    Sum the gradients $\frac{\partial J}{\partial Z_h}$ across the batch dimension.
    $$
    \nabla_{b_h} J = \sum_{i=1}^N \left(\frac{\partial J}{\partial Z_h}\right)_i \quad (\text{Sum over batch axis } 0)
    $$
    * *Dimensions: Summing $(N \times d_h)$ over axis 0 $\to (d_h)$.* (Matches $b_h$ shape)

---

### 5. Overfitting and Regularization

High-capacity networks (low bias) are prone to overfitting (high variance) by memorizing training data. Regularization adds a penalty to the loss function to constrain model complexity.

* **Early Stopping:** Stop training when performance on a *validation set* (not the training set) starts to get worse.
* **L2 Regularization (Weight Decay):** Adds a penalty proportional to the squared magnitude (Frobenius norm $||W||_F^2$) of the weights.
    $$
    J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2} \sum_{L \in \text{layers}} ||W_L||_F^2
    $$
    The gradient update for a weight $w$ becomes:
    $$
    w \leftarrow w - \alpha (\nabla_w J(\theta) + \lambda w) = (1 - \alpha \lambda) w - \alpha \nabla_w J(\theta)
    $$
* **L1 Regularization (Lasso):** Adds a penalty proportional to the absolute magnitude of the weights. This encourages **sparsity** (pushes many weights to be exactly zero).
    $$
    J_{reg}(\theta) = J(\theta) + \lambda \sum_{L \in \text{layers}} ||W_L||_1 = J(\theta) + \lambda \sum_{L,i,j} |w_{ij}|
    $$
* **Dropout:** During training, randomly set a fraction (e.g., $p=0.5$) of neuron activations in a hidden layer to zero on each forward pass. This prevents co-adaptation and forces the network to learn robust, redundant features. At test time, all neurons are used (no dropout).