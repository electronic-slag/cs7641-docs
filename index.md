---
layout: default
title: Proposal
nav_order: 1
---

# **SynergyTune: A Hybrid Deep Learning Approach to Personalized Music Recommendation Integrating Collaborative, Content, and Lyric-Based Features**

## 1. Introduction/Background

- **Objective**: Current music recommendation systems suffer from **data sparsity**, **cold start** issues, and **superficial understanding**. This section will discuss the respective advantages and disadvantages of collaborative filtering (CF) and content-based filtering (CBF). This project aims to build a new, advanced hybrid recommendation system that integrates user behavior, song content, and lyrical features.

## 2. Problem Definition

- **Objective**: To define the problems prevalent in traditional recommendation systems: **data sparsity**, **cold start** (inability to recommend for new users/songs), and a **superficial understanding** of music content.

##  3. Methods

- **Objective**: To elaborate on how the system will be constructed, presenting it as a hybrid model combining various machine learning techniques.
- Core technologies include:
	- **Unsupervised Learning**: Primarily used to process unstructured data like lyrics and extract deep semantic features. Specific methods include: using **Sentence-BERT** to generate lyric vectors, employing **LDA** for topic modeling, and utilizing **K-Means** for song clustering.
	- **Supervised Learning**: Used for final prediction and recommendation. Models planned for exploration include: **Neural Collaborative Filtering (NCF)**, **Wide & Deep models**, and potentially Graph Neural Networks (GNN).
	- **Hybridization Strategy**: Primarily adopting a **feature enhancement** approach, where all features derived from user behavior, song metadata, and lyric analysis are concatenated and jointly fed into a Deep Neural Network (DNN) for training.

## 4. Potential Results and Discussion

- **Objective**: To define the success criteria for the project, specifying which metrics will be used to evaluate model performance, and predicting the potential improvements.
- Core Evaluation Metrics:
	- **Prediction Accuracy Metrics**: If predicting ratings, **RMSE** (Root Mean Square Error) and **MAE** (Mean Absolute Error) will be used.
	- **Ranking Quality Metrics**: If generating recommendation lists, **Precision@K**, **Recall@K**, **MAP** (Mean Average Precision), and **NDCG** (Normalized Discounted Cumulative Gain) will be used.
	- **Qualitative Metrics**: Focusing on recommendation **Diversity**, **Novelty**, and the ability to mitigate the **cold start** problem.

## 5. References

- **Objective**: To list the core academic papers that form the basis of the project, demonstrating its theoretical foundation and alignment with cutting-edge research.

## 6. Proposed Timeline

- **Objective**: To be completed.

## 7. Team Members' Responsibilities

- **Objective**: To be completed.

## 8. Conclusion

- **Objective**: To summarize the entire proposal, emphasizing the project's core value, methodology, and expected contributions, leaving the reader with a clear and complete impression.
