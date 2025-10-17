# QTCO-GBC
This repository contains the official implementation of the paper "QTCO-GBC: Quantum Trust Consensus Optimization via Granular-Ball Computing for Multi-Scale Data Evaluation". 
## Method Overview
QTCO-GBC is used for multi-scale data evaluation to enhance explainability, fairness, and trust in data-driven decision-making. The algorithm consists of the following steps:

1. Multi-scale data construction: An adversarial autoencoder learns latent representations at medium and coarse levels from single-scale data to build multi-scale datasets.
2. Subgroup and group structure formation: A granular-ball generation method based on the entropy-informed justifiable granularity principle captures hierarchical structures among decision agents.
3. Trust and consensus optimization: A bi-level optimization strategy determines optimal trust relationships, while a quantum consensus optimization ensures fair resource allocation.

QTCO-GBC effectively addresses consensus optimization in multi-scale data evaluation, offering a new solution framework for decision-making problems.
## Code Structure
The experimental section of the method framework consists of eight files, while the ablation and comparative experiments include twelve files in total.

**·a1_DM_simulation.py：**
