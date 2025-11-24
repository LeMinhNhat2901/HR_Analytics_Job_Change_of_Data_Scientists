# HR Analytics: Job Change of Data Scientists

A comprehensive data analysis project exploring factors influencing data scientists' job change decisions, implemented using NumPy-based processing and visualization techniques.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Method](#method)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

## Introduction

### Problem Description
Understanding employee turnover in the data science field is crucial for HR departments and organizations investing in training programs. This project analyzes factors that influence whether a data scientist will seek new job opportunities.

### Motivation & Real-world Applications
- **HR Planning**: Help companies predict and reduce employee turnover
- **Training Investment**: Optimize training program investments
- **Recruitment Strategy**: Identify ideal candidate profiles
- **Retention Programs**: Design targeted retention strategies

### Specific Objectives
1. Analyze demographic and professional factors affecting job change decisions
2. Identify key predictors of employee turnover
3. Build predictive models using NumPy implementations
4. Provide actionable insights for HR departments

## Dataset

### Data Source
[Specify your data source - Kaggle, company data, etc.]

### Feature Description
- **enrollee_id**: Unique identifier for candidates
- **city**: City code
- **city_development_index**: Development index of the city (0-1 scale)
- **gender**: Gender of candidate
- **relevant_experience**: Relevant work experience
- **enrolled_university**: Type of university enrollment
- **education_level**: Education level of candidate
- **major_discipline**: Major discipline of study
- **experience**: Total work experience (years)
- **company_size**: Size of current employer
- **company_type**: Type of current employer
- **last_new_job**: Years since last job change
- **training_hours**: Hours of training completed
- **target**: Looking for job change (0: Not looking, 1: Looking)

### Dataset Characteristics
- **Size**: [Specify number of samples and features]
- **Missing Values**: Present in multiple features
- **Class Distribution**: [Specify target variable distribution]
- **Data Types**: Mixed (numerical and categorical)

## Method

### Data Processing Pipeline

#### 1. Data Loading (NumPy-only)
```python
# Custom CSV reader using NumPy
data = np.genfromtxt('data.csv', delimiter=',', dtype=None, encoding='utf-8')
```

#### 2. Data Validation & Cleaning
- **Validity Checks**: Range validation, type checking
- **Outlier Detection**: IQR method, Z-score method
- **Outlier Treatment**: Capping, removal (when necessary)

#### 3. Missing Value Handling
**Strategies implemented:**
- Mean/Median imputation for numerical features
- Mode imputation for categorical features
- Predictive imputation using custom NumPy models

#### 4. Normalization Techniques
**Min-Max Normalization:**
$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Log Transformation** (for skewed distributions):
$$X_{transformed} = \log(X + 1)$$

**Decimal Scaling:**
$$X_{scaled} = \frac{X}{10^j}$$
where j is the smallest integer such that max(|X_scaled|) < 1

#### 5. Standardization (Z-score)
$$X_{standardized} = \frac{X - \mu}{\sigma}$$
- Applied before gradient-based algorithms
- Ensures mean = 0, variance = 1

#### 6. Feature Engineering
- **Interaction Features**: Cross-feature relationships
- **Polynomial Features**: Non-linear transformations
- **Binning**: Categorical grouping of continuous variables
- **Encoding**: One-hot encoding for categorical variables

#### 7. Statistical Analysis
**Descriptive Statistics:**
- Mean, median, mode, variance, standard deviation
- Quartiles, percentiles

**Hypothesis Testing:**
- **H₀ (Null Hypothesis)**: [Specify based on analysis]
- **H₁ (Alternative Hypothesis)**: [Specify based on analysis]
- Tests: t-test, chi-square test, ANOVA

#### 8. Numerical Stability
- Use of log-sum-exp trick
- Handling floating-point precision
- Avoiding overflow/underflow in exponential calculations

### Machine Learning Algorithms (NumPy Implementation)

#### Logistic Regression
**Model:**
$$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

**Cost Function:**
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

**Gradient Descent:**
$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

**NumPy Implementation Strategy:**
- Vectorized operations using `np.dot()`
- Broadcasting for efficient computation
- Matrix operations instead of loops

#### K-Nearest Neighbors (KNN)
**Distance Metric (Euclidean):**
$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**NumPy Implementation:**
```python
# Vectorized distance calculation
distances = np.sqrt(np.sum((X_train[:, np.newaxis] - X_test)**2, axis=2))
```

### Evaluation Metrics (NumPy Implementation)

**Accuracy:**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:**
$$Precision = \frac{TP}{TP + FP}$$

**Recall:**
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score:**
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Cross-Validation:**
- K-Fold implementation using NumPy array indexing
- Stratified sampling for imbalanced datasets

## Installation & Setup

### Requirements
```
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
scikit-learn>=0.24.0  # Optional, for comparison only
```

### Installation Steps
```bash
# Clone repository
git clone https://github.com/yourusername/HR_Analytics_Job_Change_of_Data_Scientists.git
cd HR_Analytics_Job_Change_of_Data_Scientists

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step-by-Step Execution

#### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
- Load and inspect raw data
- Visualize distributions and relationships
- Identify missing values and outliers

#### 2. Data Preprocessing
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```
- Handle missing values
- Apply normalization/standardization
- Feature engineering
- Save processed data

#### 3. Modeling
```bash
jupyter notebook notebooks/03_modeling.ipynb
```
- Train models using NumPy implementations
- Evaluate performance
- Compare with scikit-learn baselines

### Running Individual Modules
```python
# Example: Using custom data processing module
from src.data_processing import load_data, preprocess_data
from src.models import LogisticRegressionNumPy

# Load and process data
X, y = load_data('data/raw/train.csv')
X_processed = preprocess_data(X)

# Train model
model = LogisticRegressionNumPy(learning_rate=0.01, iterations=1000)
model.fit(X_processed, y)
```

## Results

### Model Performance

| Metric | NumPy Implementation | Scikit-learn Baseline |
|--------|---------------------|----------------------|
| Accuracy | [X.XX%] | [X.XX%] |
| Precision | [X.XX%] | [X.XX%] |
| Recall | [X.XX%] | [X.XX%] |
| F1-Score | [X.XX%] | [X.XX%] |

### Key Insights

1. **Most Influential Factors:**
    - [Factor 1]: [Impact description]
    - [Factor 2]: [Impact description]
    - [Factor 3]: [Impact description]

2. **Patterns Discovered:**
    - [Pattern 1]
    - [Pattern 2]
    - [Pattern 3]

### Visualizations

[Include sample visualizations here]
- Feature importance charts
- Correlation heatmaps
- Distribution plots
- Model performance comparisons
- Confusion matrices

### Analysis & Comparison

- **NumPy vs Scikit-learn**: Performance comparison and trade-offs
- **Feature Engineering Impact**: Contribution of engineered features
- **Model Selection**: Rationale for chosen approaches

## Project Structure

```
HR_Analytics_Job_Change_of_Data_Scientists/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                          # Original dataset
│   │   └── train.csv
│   └── processed/                    # Cleaned and processed data
│       └── processed_data.npy
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and initial analysis
│   ├── 02_preprocessing.ipynb        # Data cleaning and transformation
│   └── 03_modeling.ipynb             # Model training and evaluation
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_processing.py            # NumPy-based data processing functions
│   ├── visualization.py              # Matplotlib/Seaborn plotting functions
│   └── models.py                     # Custom ML models (NumPy implementation)
└── results/
     ├── figures/                      # Saved visualizations
     └── metrics/                      # Evaluation results
```

### File Descriptions

- **data_processing.py**: NumPy-only implementations for data loading, cleaning, normalization, and feature engineering
- **visualization.py**: Functions for creating matplotlib/seaborn visualizations
- **models.py**: Custom ML algorithms implemented from scratch using NumPy
- **notebooks/**: Jupyter notebooks with step-by-step analysis and explanations

## Challenges & Solutions

### Challenge 1: Implementing Complex Operations Without Pandas
**Problem**: Reading CSV files and handling mixed data types without pandas
**Solution**: 
- Used `np.genfromtxt()` with custom dtype specifications
- Implemented custom functions for column selection and filtering using NumPy indexing

### Challenge 2: Categorical Encoding Without Scikit-learn
**Problem**: One-hot encoding categorical variables using only NumPy
**Solution**:
```python
# Custom one-hot encoding using NumPy
def one_hot_encode(arr):
     unique_vals = np.unique(arr)
     encoded = np.zeros((len(arr), len(unique_vals)))
     for idx, val in enumerate(unique_vals):
          encoded[arr == val, idx] = 1
     return encoded
```

### Challenge 3: Numerical Stability in Gradient Descent
**Problem**: Overflow/underflow in exponential calculations
**Solution**: Implemented log-sum-exp trick and gradient clipping

### Challenge 4: Efficient Matrix Operations
**Problem**: Memory limitations with large datasets
**Solution**: 
- Used NumPy broadcasting to avoid creating large intermediate arrays
- Implemented batch processing for gradient descent
- Utilized `np.einsum()` for complex tensor operations

### Challenge 5: Cross-Validation Implementation
**Problem**: Stratified k-fold without scikit-learn
**Solution**: Custom implementation using NumPy's advanced indexing and random sampling

## Future Improvements

1. **Advanced Algorithms**
    - Implement Random Forest from scratch
    - Add Support Vector Machine (SVM) implementation
    - Explore ensemble methods

2. **Feature Engineering**
    - Automated feature selection using correlation analysis
    - Polynomial feature generation
    - Text feature extraction from free-form fields

3. **Performance Optimization**
    - Parallel processing using NumPy's multithreading
    - GPU acceleration using CuPy (NumPy-compatible)
    - Memory-mapped arrays for large datasets

4. **Enhanced Analysis**
    - SHAP-like interpretability analysis using NumPy
    - Deeper statistical hypothesis testing
    - Time-series analysis of job change trends

5. **Deployment**
    - Create REST API for model predictions
    - Build interactive dashboard
    - Export to production-ready format

## Contributors

### Author Information
- **Name**: [Your Name]
- **Student ID**: [Your ID]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

### Contact
For questions, suggestions, or collaboration opportunities:
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **Issues**: [GitHub Issues](https://github.com/yourusername/HR_Analytics_Job_Change_of_Data_Scientists/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project was developed as part of [Course Name/Assignment] with strict adherence to NumPy-only data processing requirements and custom ML implementations.

### Acknowledgments
- Dataset source: [Citation]
- Inspiration: [References]
- Special thanks to: [Instructors/Collaborators]