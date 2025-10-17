# QTCO-GBC
This repository contains the official implementation of the paper "QTCO-GBC: Quantum Trust Consensus Optimization via Granular-Ball Computing for Multi-Scale Data Evaluation". 

## Method Overview
QTCO-GBC is used for multi-scale data evaluation to enhance explainability, fairness, and trust in data-driven decision-making. The algorithm consists of the following steps:

1. **Multi-scale data construction:** An adversarial autoencoder learns latent representations at medium and coarse levels from single-scale data to build multi-scale datasets.
2. **Subgroup and group structure formation:** A granular-ball generation method based on the entropy-informed justifiable granularity principle captures hierarchical structures among decision agents.
3. **Trust and consensus optimization:** A bi-level optimization strategy determines optimal trust relationships, while a quantum consensus optimization ensures fair resource allocation.

QTCO-GBC effectively addresses consensus optimization in multi-scale data evaluation, offering a new solution framework for decision-making problems.

## Code Structure
The experimental section of the method framework consists of seven files：

* **a1_DM_simulation.py:** Thirty simulated datasets were generated from the initial data by applying normal distribution.
* **a2_MSISs.py:** An encoder is employed to transform each dataset from a single-scale information system into a multi-scale information system and to learn scale weights for fusing features across multiple scales.
* **a3_ISs.py:** The fused information is obtained.
* **b1_Cluster_GB.py:** K-Means is applied for clustering, and granular-ball structures are generated to represent subgroup features.
* **c1_Trust.py:** Stable trust relationships are obtained through a bi-level optimization strategy.
* **d1_Optimize.py:** A optimization model is employed to achieve consensus among subgroups.
* **main.py:** Based on the achieved consensus, the obkectives are ranked.
  
The ablation and comparative experiments include twelve files. **E_abtion_BTOM.py, E_abtion_GB.py, and E_abtion_MSIS.py** are the codes for the ablation experiments. **E_compare1.py–E_compare9.py** are the codes for the baseline comparison methods.

## Usage
### Requirements
```
numpy 
pandas
math
torch
scikit-learn 
time
```

### Basic Usage

```
# Load datasets
read_excel_from_second_row_pandas(file_path)

# The acquisition of multi-scale information
process_single_matrix()

# Subgroup clustering
cluster_decision_makers()

# Subgroup structures
generat_granular_ball()

# The acquisition of stable trust relationships
optimize_multi_random_T()

# Consensus optimization
optimize_layer()
```

### Running Experiments
You can run the experiments from the paper by executing `main.py`.

Parameters can be adjusted, including:

`𝛼:` Sensitivity parameters in consensus measurement in `c1_Trust.py and d1_Optimize.py`.

`𝐵:` The minimum consensus increment in `c1_Trust.py`.

`ς:` The consensus thresholds in `d1_Optimize.py`.

## Experimental Results
The comparison between QTCO-GBC and other baseline comparison methods is based on the following metrics:

* Adjustment cost
* Degree of group consensus improvement
* Consensus efficiency
* Running time
