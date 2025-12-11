# K-Nearest Neighbors (KNN) Wine Classification

## Table of Contents
- [What is K-Nearest Neighbors?](#what-is-k-nearest-neighbors)
- [How KNN Works](#how-knn-works)
- [KNN in This Project](#knn-in-this-project)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for both classification and regression tasks. It's one of the simplest and most intuitive algorithms in machine learning, often called a "lazy learner" because it doesn't build an explicit model during training. Instead, it stores all training data and makes predictions based on similarity.

### Key Characteristics:
- **Non-parametric**: Makes no assumptions about the underlying data distribution
- **Instance-based**: Uses actual training instances to make predictions
- **Lazy Learning**: No training phase; all computation happens during prediction
- **Distance-based**: Relies on distance metrics to find similar samples

## How KNN Works

### Step-by-Step Process:

#### 1. **Choose the Number K**
Select the number of neighbors (K) to consider. This is a hyperparameter that significantly affects model performance.
- **Small K (e.g., 1-3)**: More sensitive to noise, may overfit
- **Large K (e.g., 20-50)**: Smoother decision boundaries, may underfit
- **Odd K**: Preferred for binary classification to avoid ties

#### 2. **Calculate Distance**
When a new data point arrives, calculate its distance to all training samples using distance metrics:

**Euclidean Distance** (most common):
```
d = √[(x₁-x₂)² + (y₁-y₂)² + ... + (n₁-n₂)²]
```

**Manhattan Distance**:
```
d = |x₁-x₂| + |y₁-y₂| + ... + |n₁-n₂|
```

**Minkowski Distance** (generalization):
```
d = (Σ|xᵢ-yᵢ|ᵖ)^(1/p)
```

#### 3. **Find K Nearest Neighbors**
Sort all training samples by distance and select the K closest ones.

#### 4. **Vote for Classification**
- Count the class labels among the K neighbors
- Assign the most frequent class to the new sample
- For regression, take the average of K neighbors' values

#### 5. **Make Prediction**
Output the predicted class (or value for regression).

### Example Visualization:
```
         *  Class A
      *     Class A
   ?        Unknown point
      o     Class B
         o  Class B

If K=3, the unknown point's 3 nearest neighbors are: 2 Class A, 1 Class B
Prediction: Class A (majority vote)
```

### Important Considerations:

**Feature Scaling**
KNN is highly sensitive to feature scales because it uses distance calculations. Features with larger ranges will dominate the distance metric.

**Solution**: Always normalize/standardize features using:
- StandardScaler: (x - mean) / standard_deviation
- MinMaxScaler: (x - min) / (max - min)

**Curse of Dimensionality**
As the number of features increases, the distance between points becomes less meaningful, degrading KNN performance.

**Computational Complexity**
- Training: O(1) - just stores data
- Prediction: O(n × d) - calculates distance to all n samples across d dimensions
- Can be slow on large datasets

## KNN in This Project

### Dataset Overview
We use the **Wine dataset** from scikit-learn:
- **178 samples** of wine
- **13 chemical features**: alcohol, malic acid, ash, alkalinity, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315, proline
- **3 classes**: Different wine cultivars (0, 1, 2)

### Implementation Pipeline

#### 1. Data Loading and Exploration
```python
from sklearn.datasets import load_wine
dataset = load_wine()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
```
Load the Wine dataset and convert to pandas DataFrame for easy manipulation.

#### 2. Feature and Target Separation
```python
X = df.drop('target', axis=1)  # 13 chemical features
y = df['target']                # Wine class labels (0, 1, 2)
```
Separate input features (X) from target labels (y).

#### 3. Feature Scaling (Critical Step!)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Why this matters**: Without scaling, features like 'proline' (range: 278-1680) would dominate over 'ash' (range: 1.36-3.23) in distance calculations, leading to poor predictions.

StandardScaler transforms each feature to have:
- Mean = 0
- Standard Deviation = 1

#### 4. Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```
- **80% training data**: 142 samples to learn patterns
- **20% testing data**: 36 samples to evaluate performance
- **random_state=42**: Ensures reproducibility

#### 5. Model Training
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```
**Why K=5?**
- Balances between overfitting (K too small) and underfitting (K too large)
- Odd number avoids ties in voting
- Works well for this 3-class problem

#### 6. Prediction
```python
y_pred = knn.predict(X_test)
```
For each test sample, the algorithm:
1. Calculates Euclidean distance to all 142 training samples
2. Finds the 5 nearest neighbors
3. Takes a majority vote among their classes
4. Assigns the winning class

#### 7. Model Evaluation
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
```

**Metrics Explained**:
- **Accuracy**: Overall correct predictions / total predictions (~94%)
- **Confusion Matrix**: Shows correct vs incorrect classifications per class
- **Precision**: Of predicted class X, how many were actually class X?
- **Recall**: Of actual class X, how many did we correctly identify?
- **F1-Score**: Harmonic mean of precision and recall

#### 8. Visualization

**Scatter Plot**: Shows how alcohol and color intensity separate wine classes
```python
plt.scatter(subset['alcohol'], subset['color_intensity'], label=f'Class {class_label}')
```

**Correlation Heatmap**: Identifies which features are strongly related
```python
corr = df.corr()
plt.imshow(corr, cmap='coolwarm')
```

### Why KNN Works Well Here

1. **Clear Class Separation**: Wine classes have distinct chemical profiles
2. **Moderate Dataset Size**: 178 samples is manageable for KNN's computational needs
3. **Multiple Features**: 13 features provide rich information for distance calculations
4. **Balanced Classes**: No severe class imbalance that would bias voting
5. **Feature Scaling Applied**: Ensures all features contribute equally to distance

### Prediction Example

Imagine a new wine sample with these properties:
```
Alcohol: 13.2%, Color Intensity: 5.5, Flavanoids: 2.8, etc.
```

The KNN algorithm:
1. Scales these features using the trained scaler
2. Calculates distance to all 142 training wines
3. Finds the 5 closest wines: perhaps 4 are Class 1, 1 is Class 0
4. Predicts: **Class 1** (majority vote: 4 out of 5)

## Results

### Model Performance
- **Accuracy**: ~94.44% (34/36 correct predictions)
- **Precision**: High across all classes (0.90-1.00)
- **Recall**: Strong performance (0.91-1.00)
- **F1-Score**: Balanced metric showing consistent performance

### Confusion Matrix Interpretation
```
Predicted →
Actual ↓     Class 0   Class 1   Class 2
Class 0        12         0         0
Class 1         1        12         0
Class 2         0         1        10
```
Only 2 misclassifications out of 36 test samples!

### Key Insights
- Alcohol content and color intensity are strong discriminators
- Features like flavanoids and proline show high correlation with wine class
- Model performs equally well across all three wine types

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Install required packages
pip install numpy pandas scikit-learn matplotlib
```

## Usage

```python
# Run the Jupyter notebook
jupyter notebook "KNN (3).ipynb"

# Or run cells individually in VS Code with Jupyter extension
```

### Experiment with Different K Values
```python
# Try different K values
for k in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"K={k}, Accuracy: {accuracy_score(y_test, y_pred):.2%}")
```

## Advantages of KNN

✅ Simple to understand and implement  
✅ No training phase (fast model creation)  
✅ Works well with multi-class problems  
✅ Naturally handles non-linear decision boundaries  
✅ No assumptions about data distribution  

## Limitations of KNN

❌ Slow prediction on large datasets  
❌ Sensitive to irrelevant features  
❌ Requires feature scaling  
❌ Memory intensive (stores all training data)  
❌ Suffers from curse of dimensionality  
❌ Sensitive to imbalanced datasets  

## Conclusion

This project demonstrates KNN's effectiveness for multi-class classification. With proper preprocessing (scaling) and appropriate K selection, KNN achieves excellent results on the Wine dataset. While it has limitations for large-scale applications, KNN remains a powerful baseline algorithm and works exceptionally well for moderate-sized datasets with clear decision boundaries.

---

**Author**: Shourya Shah 
**Date**: December 2025  
**License**: MIT
