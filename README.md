# 🩺 Diabetes Risk Prediction Project

A comprehensive data analysis and machine learning project to predict diabetes risk in patients.

## 📋 Project Description

This project uses advanced data science techniques to analyze and predict diabetes risk. It combines Exploratory Data Analysis (EDA), preprocessing, clustering, and multiple classification algorithms to create a robust prediction model.

### Main Objectives

- **Exploratory Analysis**: Understand distributions and correlations in the data
- **Preprocessing**: Clean and prepare data for modeling
- **Clustering**: Identify patient groups with similar profiles
- **Classification**: Predict diabetes risk using multiple machine learning models
- **Deployment**: Web application to use trained models

### Implemented Models

The project compares 6 classification algorithms:
- 🌳 **Decision Tree**
- 🌲 **Random Forest**
- 📈 **Gradient Boosting**
- 🚀 **XGBoost**
- 📊 **Logistic Regression**
- 🎯 **Support Vector Machine (SVM)**

## 📁 File Structure

```
diabetes-risk-prediction/
│
├── 📓 notebooks/              # Jupyter notebooks for analysis
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb # Data cleaning and preparation
│   ├── 03_Clustering.ipynb   # Clustering analysis
│   └── 04_Classification.ipynb # Model training and evaluation
│
├── 📊 data/                   # Project data
│   ├── raw_data.csv          # Raw data
│   ├── clean_data.csv        # Cleaned data (with scaling)
│   ├── clean_data_no_scale.csv # Cleaned data (without scaling)
│   └── clustered_data.csv    # Data with assigned clusters
│
├── 🤖 models/                 # Saved trained models
│   ├── best_DecisionTree_model.pkl
│   ├── best_RandomForest_model.pkl
│   ├── best_GradientBoosting_model.pkl
│   ├── best_XGBoost_model.pkl
│   ├── best_LogisticRegression_model.pkl
│   └── best_SVM_model.pkl
│
├── 📱 app/                    # Web application
│   └── app.py                # Flask/Streamlit application
│
├── 📈 reports/                # Reports and visualizations
│   ├── EDA_rapport.ipynb     # Exploratory analysis report
│   └── figures/              # Generated charts
│       ├── output1.png
│       └── output2.png
│
├── 📝 notes.ipynb            # Work notes
├── 📋 requirements.txt       # Python dependencies
└── 📖 README.md              # This file
```

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Jupyter Notebook or JupyterLab (optional, for notebooks)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/9asdaoui/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```

2. **Create a virtual environment** (recommended)
   ```powershell
   # On Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   ```bash
   # On Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is empty, install the following packages:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter imbalanced-learn joblib
   ```

### Running the Notebooks

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Execute the notebooks in order**:
   - `01_EDA.ipynb`: For exploratory analysis
   - `02_Preprocessing.ipynb`: For data preprocessing
   - `03_Clustering.ipynb`: For clustering analysis
   - `04_Classification.ipynb`: To train and evaluate models

### Running the Web Application

```bash
# From the root directory
cd app
python app.py
```

Then open your browser at the indicated address (usually http://localhost:5000 or http://localhost:8501).

## 📊 Project Pipeline

1. **Exploratory Data Analysis (EDA)** 📈
   - Distribution visualization
   - Correlation analysis
   - Outlier detection

2. **Preprocessing** 🔧
   - Data cleaning
   - Handling missing values
   - Normalization/Standardization
   - Feature engineering

3. **Clustering** 🎯
   - K-Means clustering
   - Risk group identification
   - Cluster visualization

4. **Classification** 🤖
   - Training 6 different models
   - Hyperparameter tuning with GridSearchCV
   - Handling class imbalance (RandomOverSampler)
   - Evaluation with multiple metrics (Accuracy, Precision, Recall, F1-score)

5. **Model Saving** 💾
   - Export best models in `.pkl` format
   - Ready for deployment

## 📈 Results

Models are evaluated using several metrics:
- **Accuracy**: Overall precision
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to detect all positive cases
- **F1-Score**: Harmonic mean of Precision and Recall

Detailed results are available in `notebooks/04_Classification.ipynb`.

## 🛠️ Technologies Used

- **Python** 🐍: Main language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: ML algorithms
- **XGBoost**: Advanced Gradient Boosting
- **Matplotlib/Seaborn**: Visualizations
- **Jupyter**: Interactive notebooks
- **Imbalanced-learn**: Handling class imbalance

## 👥 Author

- **9asdaoui** - [GitHub](https://github.com/9asdaoui)

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For any questions or suggestions, feel free to open an issue on GitHub.

---

**Note**: Make sure you have the necessary data in the `data/` folder before running the notebooks.
