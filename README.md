
# Music Genre Classification using Machine Learning

A comprehensive implementation of baseline and ensemble machine learning models for music genre classification, based on the paper **"SongNet: Real-time Music Classification"** by Zhang et al.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Comparison with Paper](#comparison-with-paper)
- [Future Work](#future-work)
- [References](#references)
- [Contributors](#contributors)
- [License](#license)

## 🎵 Overview

This project implements and compares various machine learning algorithms for automatic music genre classification using the Free Music Archive (FMA) dataset. The implementation focuses on traditional machine learning approaches and ensemble methods, serving as baseline comparisons to the deep learning C-RNN model proposed in the original SongNet paper.

### Key Objectives
- Implement baseline classifiers (KNN, Logistic Regression, MLP, SVM)
- Explore ensemble methods (Random Forest, Gradient Boosting, Voting Classifiers)
- Analyze the impact of hyperparameter tuning on model performance
- Provide comprehensive visualizations for data exploration and results analysis
- Compare results with the original paper's baseline models

## 📊 Dataset

**Free Music Archive (FMA) - Small Subset**

- **Total Samples**: 8,000 tracks
- **Duration**: 30-second clips per track
- **Genres**: 8 balanced classes (1,000 samples each)
  - Electronic
  - Experimental
  - Folk
  - Hip-Hop
  - Instrumental
  - International
  - Pop
  - Rock

- **Features**: 140 pre-computed MFCC (Mel-Frequency Cepstral Coefficients) features
- **Split Ratio**: 70% Training, 20% Validation, 10% Testing (stratified)

### Dataset Source
The FMA dataset is provided by:
- Defferrard et al., "FMA: A Dataset For Music Analysis"
- Available at: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)

## ✨ Features

### Data Exploration
- Genre distribution analysis
- Feature statistics and distributions
- PCA-based 2D visualization of genres
- Correlation analysis
- Before/after scaling comparisons

### Model Training
- 15 carefully selected model configurations
- Hyperparameter exploration for each model type
- Cross-validation for model evaluation
- Detailed performance metrics tracking

### Comprehensive Analysis
- Performance ranking across all models
- Category-wise comparisons
- Confusion matrix analysis
- Per-genre accuracy breakdown
- Overfitting detection
- Training time comparisons

### Visualizations
- 15+ interactive plots and charts
- Genre distribution plots
- Feature distribution histograms
- PCA clustering visualization
- Model performance comparisons
- Confusion matrices
- Hyperparameter impact analysis

## 🤖 Models Implemented

### 1. K-Nearest Neighbors (3 variants)
- k=5 with Euclidean distance
- k=5 with Manhattan distance
- k=7 with Euclidean distance

### 2. Logistic Regression (2 variants)
- C=1.0 (moderate regularization)
- C=10.0 (less regularization)

### 3. Multilayer Perceptron (3 variants)
- Single layer (100 neurons) with ReLU activation
- Single layer (100 neurons) with Tanh activation
- Two layers (100, 50 neurons) with ReLU activation

### 4. Support Vector Machine (3 variants)
- Linear kernel with C=1.0
- RBF kernel with C=1.0
- Linear kernel with C=10.0

### 5. Ensemble Methods (4 variants)
- Random Forest (100 estimators)
- Gradient Boosting (100 estimators)
- Voting Classifier (Hard voting)
- Voting Classifier (Soft voting)

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Optional for Jupyter Notebook
```bash
pip install jupyter notebook
```

### Clone Repository
```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

### Download Dataset
1. Download the FMA dataset from [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
2. Extract `tracks.csv` and `features.csv` to the project directory

## 🚀 Usage

### Option 1: Jupyter Notebook
```bash
jupyter notebook ml_miniproject_finalcode.ipynb
```

The notebook contains 18 cells organized into sections:
1. **Data Loading** (Cells 1-2)
2. **Exploratory Data Analysis** (Cells 3-5)
3. **Data Preprocessing** (Cell 6)
4. **Model Training** (Cells 7-12) - One category per cell
5. **Results Analysis** (Cells 13-18)

### Running Specific Models
Each model category can be run independently in the notebook:
- Cell 8: K-Nearest Neighbors
- Cell 9: Logistic Regression
- Cell 10: Multilayer Perceptron
- Cell 11: Support Vector Machine
- Cell 12: Ensemble Methods

## 📈 Results

### Top Performing Models

| Rank | Model | Test Accuracy | F1-Score | Training Time |
|------|-------|---------------|----------|---------------|
| 1 | Voting Classifier (Soft) | ~0.50 | ~0.49 | ~45s |
| 2 | Gradient Boosting | ~0.49 | ~0.48 | ~120s |
| 3 | Random Forest | ~0.48 | ~0.47 | ~15s |

*Note: Actual results may vary slightly due to randomization*

### Key Findings

1. **Ensemble methods** outperform individual classifiers by ~5-7%
2. **Hyperparameter tuning** significantly impacts performance (up to 10% improvement)
3. **SVM with linear kernel** performs best among traditional classifiers
4. **MLP with ReLU activation** outperforms other activation functions
5. **Manhattan distance** shows comparable performance to Euclidean in KNN

### Performance by Category

| Category | Best Accuracy | Average Accuracy |
|----------|---------------|------------------|
| Ensemble Methods | 0.50 | 0.48 |
| Support Vector Machine | 0.48 | 0.46 |
| Multilayer Perceptron | 0.46 | 0.44 |
| Logistic Regression | 0.44 | 0.43 |
| K-Nearest Neighbors | 0.42 | 0.40 |



## 📊 Visualizations

The project generates comprehensive visualizations including:

1. **Data Exploration**
   - Genre distribution bar chart
   - MFCC feature distributions
   - PCA 2D projection of genres
   - Feature correlation heatmap

2. **Model Performance**
   - Category-wise accuracy comparisons
   - Overall model ranking
   - Training time analysis
   - Overfitting analysis

3. **Best Model Analysis**
   - Confusion matrix heatmap
   - Per-genre accuracy breakdown
   - Classification report visualization

4. **Comparison Charts**
   - Paper baseline vs our results
   - Baseline vs ensemble performance
   - Hyperparameter impact analysis

## 🔬 Comparison with Paper

### Baseline Model Comparison

| Model | Paper Accuracy | Our Best Accuracy | Improvement |
|-------|----------------|-------------------|-------------|
| K-Nearest Neighbors | 36.38% | ~42% | +5.6% |
| Logistic Regression | 42.25% | ~44% | +1.8% |
| Multilayer Perceptron | 44.88% | ~46% | +1.1% |
| Support Vector Machine | 46.38% | ~48% | +1.6% |

### Performance Gap Analysis

- **Paper's C-RNN**: 65.23% accuracy (using raw mel-spectrograms)
- **Our Best Model**: ~50% accuracy (using MFCC features)
- **Gap**: ~15% 

The performance gap highlights the advantage of:
1. Deep learning architectures (CNN + RNN)
2. Raw spectrogram features vs hand-crafted MFCC features
3. End-to-end learning vs traditional feature engineering

## 🔮 Future Work

### Potential Improvements

1. **Feature Engineering**
   - Combine MFCC with other audio features (chroma, spectral contrast)
   - Experiment with different feature extraction methods
   - Include temporal dynamics features

2. **Deep Learning Implementation**
   - Implement CNN-based architectures
   - Add LSTM/GRU for temporal modeling
   - Replicate the C-RNN architecture from the paper

3. **Data Augmentation**
   - Pitch shifting
   - Time stretching
   - Adding background noise
   - Mixup techniques

4. **Advanced Ensembles**
   - Stacking classifiers
   - Blending multiple models
   - Neural network ensembles

5. **Metadata Integration**
   - Include artist information
   - Release year features
   - Album metadata

6. **Hyperparameter Optimization**
   - Grid search
   - Random search
   - Bayesian optimization

##  Acknowledgments

- Stanford CS229 Machine Learning course
- Free Music Archive dataset creators
- SongNet paper authors
- Scikit-learn community
- Open source contributors




