"""
run_xgboost_numpy_demo.py

Example script to run and test the xgboost_numpy implementation
"""
import sys 
import os 
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Add package root to path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xgboost_numpy.xgboost_numpy import GBT, ExclusiveFeatureBundler
from xgboost_numpy._base import BoosterConfig, SplitConfig

# --- Generate Data ---
X, y = make_classification(
    n_samples=1000, n_features=50, n_informative=10, n_redundant=20,
    n_classes=4, n_clusters_per_class=2, random_state=42, flip_y=0.05
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# --- Common Config ---
booster_cfg = BoosterConfig(n_estimators=50, learning_rate=0.1)

# --- 1. Exact Greedy Algorithm ---
print("\n--- 1. Training xgboost_numpy  with 'exact' algorithm ---")
split_cfg_exact = SplitConfig(max_depth=4, algorithm='exact')
model_exact = GBT(booster_cfg, split_cfg_exact)
start = time.time()
model_exact.fit(X_train, y_train)
exact_time = time.time() - start
proba_exact = model_exact.predict_probabilitys(X_test)
loss_exact = log_loss(y_test, proba_exact)

# --- 2. GOSS Algorithm ---
print("\n--- 2. Training xgboost_numpy with 'goss' algorithm ---")
split_cfg_goss = SplitConfig(max_depth=4, algorithm='goss', goss_a=0.2, goss_b=0.1)
model_goss = GBT(booster_cfg, split_cfg_goss)
start = time.time()
model_goss.fit(X_train, y_train)
goss_time = time.time() - start
proba_goss = model_goss.predict_probabilitys(X_test)
loss_goss = log_loss(y_test, proba_goss)

# --- 3. EFB + GOSS Algorithm ---
print("\n--- 3. Training xgboost_numpy with EFB + 'goss' ---")
bundler = ExclusiveFeatureBundler(max_bins=64, collision_threshold=0.2)
model_efb_goss = GBT(booster_cfg, split_cfg_goss, efb_bundler=bundler)
start = time.time()
model_efb_goss.fit(X_train, y_train)
efb_goss_time = time.time() - start
proba_efb_goss = model_efb_goss.predict_probabilitys(X_test)
loss_efb_goss = log_loss(y_test, proba_efb_goss)

# --- Print Comparison ---
print("\n\n--- Performance Comparison ---")
print(f"                                   Exact        |    GOSS         |    EFB + GOSS")
print(f"----------------------------------------------------------------------------------")
print(f"Training Time (s):               {exact_time:<12.4f} |  {goss_time:<12.4f} |  {efb_goss_time:<12.4f}")
print(f"LogLoss on test set:             {loss_exact:<12.4f} |  {loss_goss:<12.4f} |  {loss_efb_goss:<12.4f}")
print("\nNote: GOSS and EFB provide significant speedups with comparable accuracy.")

