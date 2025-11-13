## Research Notes: Module 1 - The ML Landscape & Core Trade-offs

**Objective:** To formally define the machine learning problem, differentiate its primary paradigms, and introduce the fundamental mathematical trade-off (Bias-Variance) that governs all model selection.

**References:**
* (0) Slides - Module 1 - Machine Learning The Big Picture.pdf
* (1) Slides - Module 1 - Machine Learning The Big Picture (Live Session).pdf

---

### 1. A Formal Definition of Learning

Machine Learning, at its core, is about building a program that learns from experience. A widely accepted definition from Tom Mitchell (1997) states:

> "A computer program is said to learn from **experience $E$** with respect to some class of **tasks $T$** and performance **measure $P$**, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$."

* **Task (T):** The problem to be solved (e.g., classification, regression).
* **Experience (E):** The data used for learning (e.g., a dataset $D$).
* **Performance (P):** A metric to quantify model quality (e.g., accuracy, MSE).

---

### 2. The Three Paradigms of Machine Learning

We can categorize most ML problems based on the "experience" (learning signal) available.

#### 2.1. Supervised Learning
The agent learns from a dataset of labeled examples, $D = \{(x_i, y_i)\}_{i=1}^N$.

* **Goal:** Learn a function $h: X \to Y$ that approximates an unknown true function $f(x)$. The model $h(x)$ is often called the *hypothesis*.
* **Learning Signal:** The "correct output" or *label* $y_i$ for each input $x_i$.
* **Tasks:**
    1.  **Classification:** The target $y_i$ is a discrete, categorical label. (e.g., $y_i \in \{\text{spam, not spam}\}$).
    2.  **Regression:** The target $y_i$ is a continuous, real-valued number. (e.g., $y_i \in \mathbb{R}$ representing a house price).

#### 2.2. Unsupervised Learning
The agent learns from an unlabeled dataset, $D = \{x_i\}_{i=1}^N$.

* **Goal:** Discover hidden structure, patterns, or representations in the data.
* **Learning Signal:** There are no explicit labels. The learning signal is intrinsic to the data's structure.
* **Tasks:**
    1.  **Clustering:** Group similar data points together (e.g., K-Means).
    2.  **Density Estimation:** Learn the underlying probability distribution $p(x)$ of the data (e.g., GMMs).
    3.  **Dimensionality Reduction:** Find a lower-dimensional representation $z$ of $x$ (e.g., PCA).

#### 2.3. Reinforcement Learning
The agent learns by interacting with an environment over time.

* **Experience:** A sequence of (state, action, reward) tuples: $E = (S_0, A_0, R_1, S_1, A_1, R_2, \dots)$.
* **Goal:** Learn a *policy* $\pi(a|s)$ (a mapping from states to actions) that maximizes the cumulative discounted reward (the *return*).
* **Learning Signal:** A scalar *reward signal* $R_t$, which is often sparse and delayed.

---

### 3. The Supervised Learning Pipeline

The process of building a supervised learning model generally follows this pipeline:

1.  **Data ($D$):** The raw collection of training data, $D = \{(X^{train}, y^{train})\}$.
2.  **Feature Encoding:** Raw data (e.g., images, text) is transformed into a meaningful numerical representation $\tilde{X}^{train}$.
3.  **Algorithm (Training):** The encoded features $\tilde{X}^{train}$ and labels $y^{train}$ are fed into a learning algorithm (e.g., Decision Tree, NN). The algorithm uses an **Objective Function** (or *Loss Function*) to guide its search for the best model parameters.
4.  **Model ($h(X)$):** The output of the training process is a *trained model* $h(X)$ (also called a hypothesis).
5.  **Evaluation:** The model's performance $P$ is measured by using it to make predictions $\tilde{y}^{test}$ on new, unseen test data $X^{test}$. The predictions are compared to the true labels $y^{test}$ to get a final **Score**.

---

### 4. The [[Bias-Variance Trade-off]]

This is the most important fundamental concept in model selection. It provides a mathematical decomposition of a model's expected error.

**Axiomatic Setup:**
Assume an unknown true function $f(x)$ generates our data, with added irreducible noise $\epsilon$.
$$
y = f(x) + \epsilon \quad \text{where} \quad \mathbb{E}[\epsilon] = 0, \text{Var}(\epsilon) = \sigma^2
$$
We use a training set $D$ to learn a model $\hat{f}_D(x)$. Note that our model $\hat{f}$ is a *random variable*; it is dependent on the specific (random) dataset $D$ we sampled.

**Goal:** We want to minimize the **Expected Prediction Error (EPE)** (or Mean Squared Error, MSE) on new, unseen data points at a point $x_0$:
$$
\text{EPE}(x_0) = \mathbb{E}[(y - \hat{f}(x_0))^2 \mid X=x_0]
$$
The expectation $\mathbb{E}$ is over both the sampling of $y$ (due to $\epsilon$) and the sampling of our model $\hat{f}$ (due to $D$).

**Derivation:**
We can decompose this error:
$$
\begin{aligned}
\text{EPE}(x_0) &= \mathbb{E}[(f(x_0) + \epsilon - \hat{f}(x_0))^2] \\
&= \mathbb{E}[(f(x_0) - \hat{f}(x_0) + \epsilon)^{2}] \\
&= \mathbb{E}[(f(x_0) - \hat{f}(x_0))^2] + 2\mathbb{E}[(f(x_0) - \hat{f}(x_0))\epsilon] + \mathbb{E}[\epsilon^2]
\end{aligned}
$$
Since $\hat{f}$ is built without knowledge of the future noise term $\epsilon$, and $\mathbb{E}[\epsilon]=0$, the cross-product term is zero.
$$
\text{EPE}(x_0) = \mathbb{E}[(f(x_0) - \hat{f}(x_0))^2] + \sigma^2
$$
This first term is the model's Mean Squared Error (MSE). Now, we decompose this MSE term by adding and subtracting the *average model* $\mathbb{E}[\hat{f}(x_0)]$ (the average prediction our algorithm would make at $x_0$ if trained on many different datasets $D$).
$$
\begin{aligned}
\text{MSE} &= \mathbb{E}\left[\left( (f(x_0) - \mathbb{E}[\hat{f}(x_0)]) + (\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0)) \right)^2\right] \\
&= \mathbb{E}\left[(f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2\right] + \mathbb{E}\left[(\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0))^2\right] + 2 \times \text{cross-term}
\end{aligned}
$$
The cross-term is zero. The first term is not random; it's a fixed value.
$$
\text{MSE} = \underbrace{(f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2\right]}_{\text{Variance}}
$$

**The Final Decomposition:**
$$
\text{EPE}(x_0) = (\text{Bias}(\hat{f}(x_0)))^2 + \text{Var}(\hat{f}(x_0)) + \sigma^2
$$
$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

* **Bias:** $(\mathbb{E}[\hat{f}(x_0)] - f(x_0))^2$
    * **Intuition:** The error from the model's "wrong assumptions." A simple linear model trying to fit a complex sine wave has high bias. This leads to **underfitting**.
* **Variance:** $\mathbb{E}[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2]$
    * **Intuition:** The error from the model's sensitivity to the specific training data. A high-degree polynomial that wiggles to fit every data point will change wildly with a new dataset. This leads to **overfitting**.
* **Irreducible Error:** $\sigma^2$
    * **Intuition:** The noise inherent in the data itself. This cannot be reduced by any model.

**The Trade-off:**
* Simple models (e.g., linear regression) have **high bias** and **low variance**.
* Complex models (e.g., deep neural networks, unpruned decision trees) have **low bias** and **high variance**.
* The central goal of model selection is to find the "sweet spot" of model complexity that minimizes the *sum* of these two errors.
