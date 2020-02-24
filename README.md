# Data Valuation

This repo contains the implementation of algorithms in the series of data valuation papers that we published:

1. R. Jia, D. Dao, B. Wang, F. A. Hubis, N. M. Gurel, B. Li, C. Zhang, C. Spanos, D. Song. Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms. PVLDB, 12(11): 1610-1623, 2019.

2. R. Jia*, D. Dao*, B. Wang, F. A. Hubis, N. Hynes, N. M. Gurel, B. Li, C. Zhang, D. Song, C. Spanos. Towards Efficient Data Valuation Based on the Shapley Value. AISTATS, 2019.

We provide the scripts to calculate exact Shapley value (in the `exact_sp.py`) and approximate Shapley value based on LSH (in the `LSH_sp.py`) for KNN classifier. We also provide two examples about how to calculate exact Shapley value (in the `exact_sp_example.py`) and approximate Shapley value (in the `LSH_sp_example.py`) on Cifar-10 dataset. In the reproduction folder, we provide our jupyter notebook scripts for three datasets (Cifar-10, ImageNet, and YFCC100M), which recorded our experiment results, to help reproduce our experiments. For example:
![result](result.png)

In the knn_mc folder, we implement an improved Monte Carlo approach to approximate the Shapley value for weighted KNN.

In the group_testing folder, we implelment the group testing-based approximation algorithm to approximate the Shapley value for general bounded utility functions.

In the use-case folder, we demonstrate how to apply the Shapley value to various tasks, such as data summarization, active data acquisition, noisy label detection, and watermark detection.

If you have any questions about our code, please do not hesitate to ask in the issues. Thanks!  

          
