## Research Notes: Module 2 - Non-Parametric Models

**Objective:** To understand non-parametric models, which do not make strong assumptions about the form of the true function $f(x)$. Their complexity grows with the size of the dataset.

**References:**
* (2) Slides - Module 2 - K-NN and Decision Trees.pdf

---

### 1. k-Nearest Neighbors (k-NN)

k-NN is an **instance-based** or **lazy learner**; it does no explicit "training" but instead stores the entire training set $D$. It generalizes *at prediction time*.

#### 1.1. k-NN Classification

**Algorithm:**
To classify a new test point $x_q$:
1.  **Find Neighbors:** Identify the $k$ training samples $\{x_i\}_{i=1}^k$ that are "closest" to $x_q$ based on a chosen distance metric.
2.  **Vote:** Assign the class label $y_q$ that is the *majority* among the labels $\{y_i\}_{i=1}^k$ of the $k$ neighbors.

* **Choice of $k$:** This is a critical hyperparameter that controls the bias-variance trade-off.
    * **$k=1$ (1-NN):** Creates a complex decision boundary (a Voronoi tessellation). It has zero training error, **low bias**, but **very high variance** (highly sensitive to noise).
    * **Large $k$:** Creates a smoother decision boundary, averaging over many neighbors. It has **higher bias** but **lower variance**.
    * $k$ is typically chosen via cross-validation.

* **Distance Metrics:** The definition of "closest" is a hyperparameter. For two vectors $x^{(i)}$ and $x^{(q)}$:
    * **L2 (Euclidean):** $d(x^{(i)}, x^{(q)}) = \sqrt{\sum_{j} (x_j^{(i)} - x_j^{(q)})^2}$.
    * **L1 (Manhattan):** $d(x^{(i)}, x^{(q)}) = \sum_{j} |x_j^{(i)} - x_j^{(q)}|$.
    * **Note:** k-NN is highly sensitive to feature scaling. Features with large scales (e.g., salary) will dominate features with small scales (e.g., age). Data must be normalized.

#### 1.2. Distance-Weighted k-NN
A refinement to standard k-NN that addresses the "distant neighbor" problem.

* **Algorithm:** Instead of a simple majority vote, each neighbor's vote is weighted by the *inverse* of its distance.
    $$
    w^{(i)} = \frac{1}{d(x^{(i)}, x^{(q)})^2}
    $$
    The predicted class is the one with the maximum *sum of weights* $\sum_{i=1}^k w^{(i)} \mathbb{I}(y^{(i)} = c)$ for class $c$.
* **Advantage:** This makes the choice of $k$ less critical. Large $k$ can be used, as distant neighbors will have near-zero weight.

#### 1.3. k-NN Regression
The same algorithm can be used for regression.
* **Algorithm:** To predict a value for $x_q$:
    1.  Find the $k$ nearest neighbors $\{x_i\}_{i=1}^k$.
    2.  The prediction $\hat{y}_q$ is the **average** of their values: $\hat{y}_q = \frac{1}{k} \sum_{i=1}^k y_i$.
* A distance-weighted average can also be used.

---

### 2. Decision Trees

Decision Trees are **eager learners**. They learn an explicit function $h(x)$ during training by recursively partitioning the feature space. The learned model is a tree structure, where internal nodes are "split" rules and leaf nodes are predictions.

#### 2.1. The Splitting Metric: Information Gain

How do we choose the "best" split rule at each node? We choose the split that maximizes **Information Gain**.

**1. Entropy**
First, we define **Entropy** $H(D)$ as a measure of impurity or uncertainty in a dataset $D$.
Let $p_c$ be the proportion of examples in $D$ that belong to class $c$.
$$
H(D) := - \sum_{c=1}^{K} p_c \log_2(p_c)
$$
* **Perfectly Pure Set:** (All examples are one class, $p_c=1$). $H(D) = -1 \log_2(1) = 0$. (No uncertainty).
* **Perfectly Mixed Set:** (50/50 binary classes). $H(D) = -[0.5 \log_2(0.5) + 0.5 \log_2(0.5)] = 1$ bit. (Maximum uncertainty).

**2. Conditional Entropy**
Next, we measure the *average* entropy *after* splitting dataset $D$ on an attribute $A$ into $V$ subsets $D_v$ (e.g., $v \in \{\text{sunny, overcast, rain}\}$).
$$
H(D \mid A) := \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)
$$
This is the weighted average of the remaining entropy.

**3. Information Gain (IG)**
Information Gain is the **reduction in entropy** achieved by splitting on attribute $A$.
$$
\text{I}(D, A) := H(D) - H(D \mid A)
$$
$$
\text{I}(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)
$$
The best split attribute $A^*$ is the one that maximizes $\text{IG}$.

#### 2.2. The CART/ID3 Algorithm (Conceptual)

The tree is built top-down using a greedy search.

1.  Start with the full dataset $D$ at the root node.
2.  **Find Best Split:** For *every* feature $f$ and *every* possible split point $v$:
    * Calculate the $\text{I}(D, \text{split on } f,v)$.
3.  **Make Split:** Select the feature and split point $(f^*, v^*)$ that yields the highest Information Gain.
4.  **Recurse:**
    * Create a child node for each resulting subset $D_v$.
    * Repeat from step 1 on each child node with its subset $D_v$.
5.  **Stop:** Stop recursion when:
    * A node is 100% pure (all examples have the same class).
    * A pre-defined `max_depth` is reached.
    * The number of examples in a node is below `min_samples_split`.
    * The Information Gain from splitting is below a `gain_threshold`.
    A leaf node predicts the majority class of the examples it contains.

#### 2.3. Overfitting in Decision Trees

A fully-grown tree will have **low bias** (it can memorize the training data perfectly) but **extremely high variance** (it's highly sensitive to the training data). This is overfitting.

**How to Control Overfitting:**
1.  **Early Stopping (Pre-pruning):** Stop the tree from growing by tuning hyperparameters (e.g., `max_depth`, `min_samples_split`).
2.  **Pruning (Post-pruning):**
    * Grow the full (overfit) tree.
    * Iterate from the bottom up. For each internal node, check if replacing its sub-tree with a single leaf node *improves* accuracy on a separate **validation set**.
    * If accuracy improves (or doesn't hurt much), "prune" the sub-tree.

#### 2.4. Regression Trees
Decision trees can also be used for regression. The algorithm is identical, but two things change:
* **Split Metric:** Instead of Information Gain (entropy), we use **Variance Reduction**. The best split is the one that most decreases the weighted average variance of the $y$ values in the resulting subsets.
    $$
    \text{Var}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y})^2 \quad \text{where} \quad \bar{y} = \frac{1}{|D|} \sum_{i \in D} y_i
    $$
    $$
    \text{VarReduction}(D, A) = \text{Var}(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} \text{Var}(D_v)
    $$
* **Leaf Prediction:** A leaf node predicts the **average (mean)** of the $y$ values of the training samples in that leaf.