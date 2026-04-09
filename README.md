# Ordinal-aware-Contrastive-Learning-for-Diabetic-Retinopathy-Grading
Ordinal-aware contrastive learning framework for diabetic retinopathy grading, leveraging label ordering to improve classification and representation learning.

Overview

Diabetic Retinopathy (DR) grading is inherently an ordinal classification problem, where disease severity progresses in ordered stages. 
Conventional classification methods often ignore this structure, leading to suboptimal feature learning and misclassification between adjacent grades.
This repository presents an ordinal-aware contrastive learning framework that explicitly incorporates label ordering into representation learning.
The approach combines contrastive learning with ordinal constraints, along with multi-objective optimization (CE + MSE + Ordinal loss), to produce robust and structured feature embeddings for DR grading.

Key Contributions
* Ordinal-aware contrastive learning to enforce structured feature distances
* Multi-loss framework integrating:
  * Cross-Entropy (CE)
  * Mean Squared Error (MSE)
  * Ordinal Loss
* EMA (Exponential Moving Average) for stable training
* Class Balanced MoCo design
* Model ensembling using EMA weights across epochs
