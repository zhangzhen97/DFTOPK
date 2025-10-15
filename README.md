# DFTopK: Differentiable Fast Top-K Selection for Large-scale Recommendation

[![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-green)](https://github.com/RecFlow-nips24/RecFlow-nips24/blob/main/LICENSE)

---

## 📘 Overview

Cascade ranking is a predominant architecture in large-scale information systems for top-K selection. However, the non-differentiable nature of sorting prevents direct end-to-end(E2E) optimization. Prior work has addressed this challenge through two main approaches: 
(1) Learning-to-Rank (LTR) losses optimized for ranking objectives, 
and (2) Top-K losses based on differentiable permutation matrices. 
The former lacks explicit Top-K objective modeling, while the latter achieves E2E optimization at the cost of gradient conflicts and $O(n^2)$ complexity.
A natural alternative involves soft Top-K surrogates via differentiable operators such as LAPSUM.
However, even the best operators require $O(n log n)$ or worse complexity, falling short of optimal linear time.
We propose DiffOnTopK, a novel differentiable operator with optimal O(n) complexity. By strategically relaxing normalization constraints, it provides a closed-form solution that eliminates sorting and iterative computation. Our design not only achieves theoretical optimal efficiency but also effectively mitigates the gradient conflicts inherent in permutation-based methods.We validate our contributions with extensive experiments, first by benchmarking our operator's efficiency and effectiveness on public datasets, and then by deploying it in a real-world advertising system. 
The results show that our operator's efficiency gains can be directly translated into performance improvements by scaling up the training data, leading to significant business value.
To the best of our knowledge, our work is the first to introduce differentiable Top-K operators to the field of recommendation, and also proposes the first such operator with theoretically optimal time complexity. We have open-sourced the relevant code to support future work in this direction from both academia and industry.

---

## 🔁 How to Reproduce Experiments on RecFlow

### 📁 Data Preparation

To run the experiments, please download the dataset from [this link](https://rec.ustc.edu.cn/share/883adf20-7e44-11ef-90e2-9beaf2bdc778). After downloading, organize the data under the `./data/` directory according to the expected structure.

---

### 🧪 Code Execution

Below are example commands to reproduce the experiments in this paper.

#### For main results

```bash

# NeuralSort baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "lcron_v1" 0 1 "31" 8 0 "neural_sort"

# DiffSort baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "lcron_v2" 0 1 "31" 8 0 "diff_sort"

# SoftSort baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "lcron_v3" 0 1 "31" 8 0 "soft_sort"

# DFTopK baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "cascade-topk_dftopkjoint" 0 500 "31" 8 0

# Google Sparse TopK baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "cascade-topk_googlejoint" 0 1 "31" 8 0

# Lapsum baseline
bash ./two_stage/run_x2_negativesampling.sh "all" "cascade-topk_lapsumjoint" 0 1 "31" 8 0

```


#### Runtime Analysis

```bash
# GPU Runtime
bash ./two_stage/count_time_gpu.sh

# CPU Runtime
bash ./two_stage/count_time_cpu.sh
```

## ⚙️ Requirements
The code has been tested under the following environment:

```yaml
python=3.7
numpy=1.18.1
pandas=1.3.5
pyarrow=8.0.0
scikit-learn=1.0.2
torch=1.13.1
faiss-gpu=1.7.1
```

> **Note:** We recommend running these experiments on A800 GPUs.