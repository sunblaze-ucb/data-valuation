# Data Valuation

This repo contains scripts to calculate exact Shapley value (in the `exact_sp.py`) and approximate Shapley value based on LSH (in the `LSH_sp.py`) for KNN classifier.

We also provide two examples about how to calculate exact Shapley value (in the `exact_sp_example.py`) and approximate Shapley value (in the `LSH_sp_example.py`) on Cifar-10 dataset.

In the reproduction folder, we provide our jupyter notebook scripts for tree datasets (Cifar-10, ImageNet, and YFCC100M), which recorded our experiment results, to help reproduce our experiments.

For example:
![result](result.png)

In the knn_mc folder, we implement an improved Monte Carlo approach to approximate the Shapley value for weighted KNN.

In the use-case folder, we demonstrate the use of the Shapley value in various tasks, such as data summarization, active data acquisition, noisy label detection, and watermark detection.

In the group_testing folder, we implelment the group testing-based approximation algorithm to approximate the Shapley value for generall bounded utility functions.

If you have any questions about our code, please do not hesitate to ask in the issues. Thanks!  

          
