# DFTopK: Differentiable Fast Top-K Selection for Large-scale Recommendation

[![License](https://img.shields.io/badge/license-CC%20BY--SA%204.0-green)](https://github.com/RecFlow-nips24/RecFlow-nips24/blob/main/LICENSE)

---

## 📘 Overview

Cascade ranking is a widely adopted paradigm in large-scale information retrieval systems for Top-K item selection. However, the Top-K operator is non-differentiable, hindering end-to-end training. Existing methods include Learning-to-Rank approaches (e.g., LambdaLoss), which optimize ranking metrics like NDCG and suffer from objective misalignment, and differentiable sorting-based methods (e.g., ARF, LCRON), which relax permutation matrices for direct Top-K optimization but introduce gradient conflicts through matrix aggregation. A promising alternative is to directly construct a differentiable approximation of the Top-K selection operator, bypassing the use of soft permutation matrices. However, even state-of-the-art differentiable Top-K operator (e.g., LapSum) require O(nlogn) complexity due to their dependence on sorting for solving the threshold. Thus, we propose DFTopK, a novel differentiable Top-K operator achieving optimal O(n) time complexity. By relaxing normalization constraints, DFTopK admits a closed-form solution and avoids sorting. DFTopK also avoids the gradient conflicts inherent in differentiable sorting-based methods. We evaluate DFTopK on both the public benchmark RecFLow and an industrial system. Experimental results show that DFTopK significantly improves training efficiency while achieving superior performance, which enables us to scale up training samples more efficiently. In the online A/B test, DFTopK yielded a +1.77\% revenue lift with the same computational budget compared to the baseline. To the best of our knowledge, this work is the first to introduce differentiable Top-K operators into recommendation systems and the first to achieve theoretically optimal linear-time complexity for Top-K selection. We have open-sourced our implementation to facilitate future research in both academia and industry.

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

## 📎 Cite this Work

If you find this project helpful for your research, please cite our paper:

```
@misc{zhu2025differentiablefasttopkselection,
      title={Differentiable Fast Top-K Selection for Large-Scale Recommendation}, 
      author={Yanjie Zhu and Zhen Zhang and Yunli Wang and Zhiqiang Wang and Yu Li and Rufan Zhou and Shiyang Wen and Peng Jiang and Chenhao Lin and Jian Yang},
      year={2025},
      eprint={2510.11472},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.11472}, 
}
```
