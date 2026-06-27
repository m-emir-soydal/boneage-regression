"""Fair multi-seed UQ pipeline (BNN + MC-Dropout) for head-to-head comparison
with the conformal-prediction methods (SCP / KNN-NCP / AS-MCP).

Every seed reproduces the *exact* per-seed data partition used by the conformal
pipeline (stratified 50/25/25 split of ``train.csv`` with ``random_state=seed``,
external ``test.csv`` as the held-out test set), so for matching seeds the
BNN/MC-Dropout models and the conformal models see byte-identical data.
"""
