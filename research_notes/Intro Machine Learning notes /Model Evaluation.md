## Research Notes: Module 3 - Model Evaluation

**Objective:** To understand *how* to measure the performance $P$ of a learned model $h(x)$ and how to reliably estimate its ability to generalize to new, unseen data.

**References:**
* (3) Slides - Module 3 - Evaluation of Machine Learning Systems.pdf

---

### 1. The Problem: Evaluating Generalization

* **Training Error:** The error of a model on the same data it was trained on. This is a poor (overly optimistic) estimate of future performance.
* **Generalization Error (Test Error):** The error of a model on new, unseen data. This is the true measure of a model's quality.
* **Goal:** We want to estimate the generalization error.

---

### 2. Data Splitting Strategies

#### 2.1. Train-Test Split
The simplest strategy.
1.  Shuffle the dataset $D$.
2.  Split it into $D_{train}$ (e.g., 80%) and $D_{test}$ (e.g., 20%).
3.  **Train:** Learn the model $h$ using only $D_{train}$.
4.  **Test:** Evaluate $h$ on $D_{test}$. The error on $D_{test}$ is our estimate of the generalization error.

#### 2.2. Train-Validation-Test Split
This is the standard for **hyperparameter tuning** (e.g., choosing $k$ in k-NN or `max_depth` in a Decision Tree).

1.  Split $D$ into $D_{train}$ (e.g., 70%), $D_{val}$ (e.g., 15%), and $D_{test}$ (e.g., 15%).
2.  **Training Loop:** For each candidate hyperparameter $\lambda$ (e.g., $\lambda \in \{1, 5, 10, 20\}$):
    a. Train a model $h_\lambda$ on $D_{train}$.
    b. Evaluate $h_\lambda$ on $D_{val}$ to get a score $P_\lambda$.
3.  **Model Selection:** Choose the $\lambda^*$ that gave the best score $P_{\lambda^*}$ on the validation set.
4.  **Final Evaluation:** Train a new model $h_{\lambda^*}$ on the *combined* $D_{train} + D_{val}$.
5.  **Report Score:** Report the final, unbiased performance of $h_{\lambda^*}$ on $D_{test}$. **The test set is only used once.**

#### 2.3. k-Fold Cross-Validation (CV)
Used when the dataset $D$ is small, as it provides a more robust estimate of generalization error.

1.  Shuffle $D$ and split it into $k$ equal-sized, non-overlapping folds (e.g., $k=10$).
2.  Loop $i$ from $1$ to $k$:
    a. **Train:** Train model $h_i$ on all folds *except* fold $i$.
    b. **Validate:** Test $h_i$ on the held-out fold $i$ to get score $P_i$.
3.  **Final Score:** The final estimate of generalization error is the average of the scores from all $k$ folds: $P_{CV} = \frac{1}{k} \sum_{i=1}^k P_i$.

---

### 3. Evaluation Metrics for Classification

Once we have predictions $\hat{y}$ and true labels $y$ for a test set, we need a metric $P$ to score performance.

#### 3.1. The Confusion Matrix
For a binary (Positive/Negative) classification problem, the **Confusion Matrix** is the foundation:

| | **Predicted Positive** | **Predicted Negative** |
| :--- | :--- | :--- |
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

#### 3.2. Core Metrics
From the confusion matrix, we can derive key metrics:

* **Accuracy:** The fraction of *all* predictions that were correct. Best for balanced datasets.
    $$
    \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
    $$
* **Precision (Positive Predictive Value):** Of all the times the model predicted *Positive*, what fraction was correct? Measures "don't be wrong."
    $$
    \text{Precision} = \frac{TP}{TP + FP}
    $$
* **Recall (Sensitivity, True Positive Rate):** Of all the *actual* Positive cases, what fraction did the model find? Measures "don't miss anything."
    $$
    \text{Recall} = \frac{TP}{TP + FN}
    $$
* **F1-Score:** The harmonic mean of Precision and Recall. A balanced measure that is useful for imbalanced datasets.
    $$
    F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
    $$

#### 3.3. ROC and AUC
Many models (e.g., Logistic Regression) output a *probability* $p(y=1|x)$, not a hard label. We get a label by applying a threshold (e.g., if $p > 0.5$, predict Positive).

* **Receiver Operating Characteristic (ROC) Curve:** A 2D plot that shows the trade-off between **True Positive Rate (Recall)** and **False Positive Rate (FPR)** as this decision threshold is varied from 0 to 1.
    * **Y-axis:** True Positive Rate (TPR) = $\frac{TP}{TP+FN}$
    * **X-axis:** False Positive Rate (FPR) = $\frac{FP}{FP+TN}$
    * A random classifier (coin flip) produces a diagonal line. A perfect classifier goes to the top-left corner (TPR=1, FPR=0).

* **Area Under the Curve (AUC):** The single-scalar summary of the ROC curve.
    * AUC = 1.0: Perfect classifier.
    * AUC = 0.5: Random classifier.
    * AUC < 0.5: Worse than random.
    * AUC represents the probability that a randomly chosen Positive sample is ranked higher (given a higher score) than a randomly chosen Negative sample.

---

### 4. Evaluation Metrics for Regression

For regression tasks, we measure the error between continuous true values $y_i$ and predicted values $\hat{y}_i$.

* **Mean Squared Error (MSE):** The average of the squared errors. Penalizes large errors heavily.
    $$
    \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
    $$
* **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\text{MSE}}$. Puts the error back into the original units of $y$.
* **Mean Absolute Error (MAE):** The average of the absolute errors. More robust to outliers than MSE.
    $$
    \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
    $$