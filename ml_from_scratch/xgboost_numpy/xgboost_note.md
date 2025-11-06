
*Ref :* 
	[1] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
    [2] G. Ke et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree


**The core idea of XGBoost is to formulate tree building as an optimization problem where the objective function is additive and regularized.**


### The Objective Function

Let our prediction at iteration t for instance $i$ be $y^{​i}(t)$​. We add a new function (a tree), ft​, to our previous prediction: $\hat y_{​i}^{(t)}​=\hat y_{​i}^{(t−1)}​+f_{t}​(x_{i}​)$

Our goal is to find the $f_{t}​$that minimises the overall objective:
$$
\mathcal{L}^{(t)} =  \sum_{i=1}^{n} l(y_{i}, \hat y_{i}^{(t)}) +  \sum_{k=1}^{t} \Omega (f_{k})
$$

where l is our loss function (e.g., [[LogLoss]]) and Ω is a [[regularization]] term.

The key insight is to use a second-order Taylor expansion for the loss function l around the prediction $\hat y_{i}^{(t-1)}$:
$$
l(y_{i}, \hat y_{i}^{(t-1)} + f_{t}( \mathbf{x}_{i})) \approx l(y_{i}, \hat y_{i}^{(t-1)}) + g_{i} \cdot f_{t}( \mathbf{x}_{i}) + \frac{1}{2}h_{i} \cdot f_{t}( \mathbf{x}_{i}) ^{2}
$$
where $g_{i}$ and $h_{i}$ are teh first and second derivaties ( **gradient and hessian**) 

Since $l(y_{i}\hat y_{i}^{(t-1)})$ is constanst as step $t$ , we remove it and th objective is simplieded to 
$$
\mathcal{L} ^{(t)}\approx  \sum_{i=1}^{n} \left[ g_{i}\cdot f_{t}( \mathbf{x}_{i)} + \frac{1}{2} h_{i} \cdot f_{t}( \mathbf{x}_{t})^{2}\right]+ \Omega(f_{t}) 
$$

#### Generalising to Multi-class classification

For a multi-class problem with $K$ classes, the model's output for an instance $i$ becomes a vector of logits, $\hat{\mathbf{y}}_i \in \mathbb{R}^K$.

Probabilities are derived via the **Softmax function**:

$$p_{ik} = \text{softmax}(\hat{\mathbf{y}}_i)_k = \frac{e^{\hat{y}_{ik}}}{\sum_{j=1}^K e^{\hat{y}_{ij}}}$$

The loss is the **Multi-class LogLoss** (Cross-Entropy), where the true label $y_i$ is one-hot encoded as $\mathbf{y}_i$:

$$l(\mathbf{y}_i, \hat{\mathbf{y}}_i) = - \sum_{k=1}^K y_{ik} \log(p_{ik})$$

The gradient and hessian are now class-specific:

- **Gradient:** $g_{ik} = \frac{\partial l}{\partial \hat{y}_{ik}} = p_{ik} - y_{ik}$
    
- **Hessian (Diagonal Approximation):** The full Hessian is a $K \times K$ matrix. For tractability, GBDTs use only its diagonal elements: $h_{ik} \approx \frac{\partial^2 l}{\partial \hat{y}_{ik}^2} = p_{ik}(1 - p_{ik})$
    

This approximation is critical: it decouples the problem, allowing us to build $K$ **independent trees at each boosting iteration**, one for each class, using the corresponding $g_{ik}$ and $h_{ik}$ values.


### Optimal Leaf Weights 
A tree $f_{t}$ is defined s.t 
$$
f_{t}: \mathbf{x}_{i} \mapsto w _{q(\mathbf{x}_{i}) } 
$$
and we have the regularziation defined as 
$$
\Omega(f_{t})  = \gamma \cdot T  + \frac{1}{2}\lambda \cdot \| \mathbf{w} \|^{2} = \gamma \cdot T + \frac{\lambda}{2} \cdot  \sum_{i=1}^{T} w_{i}^{2}
$$
Then we defined $I_{j}= \{ i | q(\mathbf{x}_{i}) = j\}$ the set of points that map to leaf $j$ then 
$$
\mathcal{L}^{(t)}\approx  \sum_{j=1}^{T} \left[G_{j}w_{j} + \frac{1}{2}( H_{j} + \lambda) w_{j}^{2} \right] + \gamma \cdot  T
$$
with $G_{i} =  \sum_{i \in I_{j}}^{} g_{i}$ and $H_{j} =  \sum_{i \in I_{j}}^{} h_{i}$.
Which can taking the partial with $w_{j}$ gives 
$$
w_{j}^{*} = - \frac{G_{j}}{ H_{j}+ \lambda}
$$

#### Split Gain Formula 

Subbing this back in to the loss gives minium loss which we call **qualitly score** 
$$
\mathcal{L}^{(t)}(q)  = - \frac{1}{2}  \sum_{j=1}^{T} \frac{G_{j}^{2}}{ H_{j} + \lambda} + \gamma \cdot T 
$$
We want to split the tree to maximise the Gain. 
Using similar logic as the Assingment deicision tree wanting to reduce the shared Entropy of the Left and Right nodes. 

$$
\begin{align*}
Gain &=  Loss_{Parent} - (Loss_{Left} + Loss_{Right}) \\
&= \frac{1}{2} \left[ \frac{G_{L}^{2}}{H_{L} + \lambda} + \frac{G_{R}^{2}}{ H_{R} + \gamma} - \frac{G_{P}^{2}}{H_{P} + \gamma}\right] - \gamma
\end{align*}
$$

Note that we can have less computation since 
$$
G_{P} = G_{L} + G_{R}
$$
and similary for $H$. 
### Optimisation from Light GBM 

lightGMB confront two main bottle nechs of the exact greddy algorithim above: 
1. **Cost of scanning instance**: iterating over all $n$ instance for every split is slow 
2. **Cost of scanning feaures:** iterating over all $d$ features is slow, especially for high-$d$ spares data. 

#### Gradient-based One-Side Sampling (GOSS)
***Idea**: instance with small gradients $|g_{i}|$ are already well-trained. The main Errors come from large gradients.\

**Algorithm**:
1. Sort the instances $I$ at current node by $|g_{i}|$
2. Defined a *Large gradient* set $A$ containg the top $a \%$ of instances. 
3. Defined small gradients as $A^{c} = I - A$
4. Randomly sampla a subset $B \subset A^{c}$ of size $b \times |A^{c}|$
5. Use the instance set $A \cup B$ to find the best split
6. **Crucially**: To maintain an unbiased estimate of the gain, the grdients and hessian of the set $B$ must be re-weighted by a factor $\frac{1-a}{b}$ (**inverse probability weight**.)


#### Exclusive Feature Bundling (EFB) 
***idea**: In sparse datasets (*one-hot encodings*), many feaures are *mutually exclusive*. we can bundle these feautes into a single feature, reducing the effective dimension $d$.

**Algorithm**
1. *Identify Bundles*. Model features as a graph, where an edge exists between two features if they **collide** )are non-zero simultaneosly) more than a threshold $\kappa$ , Use a [[greedy graph colouring algorithm]] to group features into bundles that are mostly exclusing. 
2. *Merge Bundles* for features $F_{1,}F_{2,}, \dots$ create a new single featur $F_{bundle}$ and add distinf offset to their binned value. 
	 - Example: F1​ (binned) is in $[0,10)$, F2​ (binned) is in $[0,20)$.
	- Fbundle​=F1​+(F2​+10).
	- A value of 8 in Fbundle​ must be from F1​.
	- A value of 25 (=15+10) in Fbundle​ must be from F2​ (with original value 15).
3. *Build Histogram* Build histograms on $F_{bundle}$. 

This reducces the complexity of the histogram building from $\mathcal{O}( n \cdot d )$ to $\mathcal{O}( n \cdot d_{bundle})$



### The Core Logic of Boosting: Why We Build So Many Trees

Let's first clarify a crucial point from your question. We are **not** making one tree for each feature. This is a common point of confusion. The process is much more subtle and powerful.

For our multi-class problem with K classes (e.g., K=4), at each step (or "iteration") of the boosting process, we build **K separate trees—one for each class**. The purpose of the tree for class k is to learn how to improve the logit score $y^{​ik}$​ for that specific class.

The overall process is called **boosting**. Think of it as building a "team of specialists."

1. **Start with a Simple Guess:** We begin not with a tree, but with a very simple, naive prediction for the logits of each class. This is our `initial_prediction`. It's a weak model, often just based on the overall class frequencies.
    
2. **Identify the Errors:** We calculate how wrong this initial prediction is. In our framework, the "error" is captured by the gradient, $g_{ik​}=p_{ik}​−y_{ik}$​. A large negative gradient for the true class means our prediction for that class is far too low. A large positive gradient for a wrong class means our prediction for that class is far too high.
    
3. **Train a Specialist to Fix the Errors:** Now, we build our first set of K trees. **The crucial insight is that these trees are not trained to predict the original labels y. Instead, each tree is trained to predict the _negative gradient_ (the error or "residual") for its specific class.** The tree learns rules like, "For instances in this region of the feature space, the model's prediction for class 2 is consistently too low, so we need to add a positive value to its logit."
    
4. **Make a Small Correction:** We don't trust any single specialist (tree) completely. We take the prediction from each new tree and add only a small fraction of it (controlled by the `learning_rate`) to our overall prediction. This is the update step: $y^{​(t)}=y^{​(t−1)}+η⋅f_{t}​(\mathbf{x})$.
    
5. **Repeat:** Now we have a slightly better model. But it still has errors. So, we re-calculate the gradients based on our _new, improved_ predictions and repeat the process. We build another set of K specialist trees to correct the _new_errors.
    

We do this `n_estimators` times. The final prediction is the sum of the initial guess plus all the small, incremental corrections made by every specialist tree we've built. This is why it's an **additive model**. Each new tree adds a small piece of knowledge, gradually refining the prediction from a simple initial guess into a highly accurate and complex function.

---

### A Deep Dive into Hyperparameters: Controlling the Model

These parameters are the levers we pull to control the model's behavior, primarily managing the **bias-variance trade-off**. High bias means the model is too simple and underfits. High variance means the model is too complex and overfits the training data.

#### Booster-Level Parameters (`BoosterConfig`)

- **`n_estimators`**:
    
    - **Purpose:** The total number of boosting rounds. For multi-class, this means we will build `n_estimators * n_classes` trees in total.
        
    - **Effect:** Increasing this number allows the model to make more corrections and fit the data more closely, reducing bias. However, too many estimators will eventually start fitting the noise in the training data, increasing variance (overfitting).
        
    - **Typical Values:** `100` to `2000`. It has a direct trade-off with `learning_rate`.
        
- **`learning_rate` (η):**
    
    - **Purpose:** A scaling factor for the contribution of each new tree. It controls the step size we take to correct the errors.
        
    - **Effect:** A lower learning rate makes the model more robust and less likely to overfit. It requires more `n_estimators` to achieve the same level of training error but often results in better generalization to unseen data. A high learning rate can cause the model to converge too quickly to a suboptimal solution.
        
    - **Typical Values:** `0.01` to `0.3`. A common strategy is to start with a low `learning_rate` (e.g., `0.05`) and find the optimal `n_estimators` using early stopping.
        

#### Tree-Level Parameters (`SplitConfig`)

These control the complexity of _each individual tree_ built at each step.

- **`max_depth`**:
    
    - **Purpose:** The maximum depth of any single tree. A tree of depth d can have at most 2d leaves.
        
    - **Effect:** This is a primary lever for controlling model complexity. Deeper trees can capture more complex and higher-order feature interactions, but they are very prone to overfitting. Shallow trees (e.g., depth 3-5) are much weaker individually but often lead to better generalization when combined in a boosting ensemble.
        
    - **Typical Values:** `3` to `8`.
        
- **`lambda_` (λ):**
    
    - **Purpose:** L2 regularization term on the leaf weights. It is added to the denominator of the optimal weight calculation: wj∗​=−Gj​/(Hj​+λ).
        
    - **Effect:** Increasing λ forces the leaf weights to be smaller (closer to zero). This makes each tree's contribution more conservative and smooths the final prediction function, reducing overfitting. A value of `0`means no regularization.
        
    - **Typical Values:** `0` to `10`. `1.0` is a very common default.
        
- **`gamma` (γ):**
    
    - **Purpose:** A pseudo-regularization hyperparameter that defines the minimum loss reduction (gain) required to make a further partition on a leaf node.
        
    - **Effect:** It's a pruning parameter. A split is only made if its gain (calculated by our formula) is greater than γ. A higher `gamma` value makes the algorithm more conservative, leading to simpler trees with fewer leaves.
        
    - **Typical Values:** `0` to `20`. `0` means no pruning based on gain.
        
- **`min_hessian`** (often called `min_child_weight` in libraries):
    
    - **Purpose:** Sets the minimum sum of hessian values allowed in a child node (a leaf).
        
    - **Effect:** Since the hessian h=p(1−p) is related to the number of instances in a node, this parameter effectively controls the minimum number of samples a leaf must contain. It prevents the creation of leaves that are based on too few instances, which would make the leaf weight estimate very noisy and lead to overfitting.
        
    - **Typical Values:** `1` to `100`. The default of `1.0` is a good starting point.
