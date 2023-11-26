# EEG Data Analysis and Machine Learning

## Overview

This project explores the application of machine learning techniques, specifically focusing on EEG data related to confusion states. The primary objectives include feature selection, model training, and the investigation of uncertainty sampling.

## Report Summary

### Objective

Real-world applications often require extensive training of machine learning algorithms. Human in the loop (HITL) machine learning integrates human expertise into algorithmic processes to enhance model accuracy and ensure ethical decision-making.

### Proposed Approach

1. **Uncertainty Sampling Approach for Training:**
   - Concept: Prioritize training on samples with the highest uncertainty.
   - Implementation: Focus on instances where the model exhibits low confidence in predictions.
   - Benefits: Enhances model robustness by improving accuracy in challenging cases.

2. **Removal of Unnecessary Features while Maintaining Accuracy:**
   - Rationale: Eliminate irrelevant features to streamline training and improve efficiency.
   - Feature Selection Criteria: Remove features with the lowest correlation to the output variable.
   - Benefits: Reduces dimensionality, potentially improving interpretability and computational efficiency.

### Dataset

- [EEG Dataset](https://www.kaggle.com/datasets/wanghaohan/confused-eeg)

### Models Considered

1. Random Forest Classifier
2. SVM Model
3. Boosted Trees (XGBoost)

### Baseline Model

- Random Forest Classifier with Random Sampling

### Experimental Findings

1. **Random Forest Model (Baseline):**
   - Initial Accuracy: 0.6384
   - Feature Removal Impact: Positive effect with the removal of the first feature.
   - Threshold Application: Effective in simplifying input space while maintaining an acceptable accuracy.

2. **Boosted Trees Model:**
   - Initial Accuracy: 0.8744
   - Feature Removal Impact: Varying effects; careful consideration needed for threshold application.

### Uncertainty Sampling Analysis

- Demonstrated notable enhancement in model accuracy, focusing on uncertain cases over random sampling.

### Future Improvements

1. Integration of Approaches: Explore synergies between Uncertainty Sampling and Feature Reduction.
2. Fine-Tuning Uncertain Sample Training: Investigate optimal strategies for determining the number of uncertain samples.
3. Testing on Diverse Machine Learning Models: Extend experimentation to a broader range of models.
4. Incorporate Domain-Specific Knowledge: Integrate domain insights into sampling strategies.
5. Ensemble Techniques: Explore benefits of ensemble techniques combining different sampling approaches.

## Python Script - MLModel Class

The provided Python script defines an `MLModel` class, encapsulating various machine learning operations on EEG data.

- **Setup:** Downloads and extracts EEG data, sets up a DataFrame.
- **Model Training:** Supports Random Forest, SVM, and Boosted Trees.
- **Data Manipulation:** Allows splitting, feature removal, and evaluating performance after removal.
- **Correlation and Visualization:** Includes methods for plotting correlation matrices and accuracies.
- **Analysis Loop:** Conducts analyses on different model performances and feature removals.
- **Uncertainty Sampling Analysis:** Investigates the impact of uncertainty sampling on model accuracy.
- **Difference in Sampling Approach:** Compares uncertainty sampling with random sampling.

## Usage

1. **Setup:**
   ```python
   m = MLModel()
   m.setup('data')

2. Toggle the False to True to see the implementation of different models and classes available in the file for feature removal and comparison.

3. A makefile is provided, run make main to download, setup and test the code.