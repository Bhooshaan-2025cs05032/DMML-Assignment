# DMML Assignment: MLFlow Experiment Tracking with DagsHub

## Project Overview

This assignment demonstrates **MLFlow experiment tracking** and **DagsHub-based centralized MLflow tracking** for collaborative machine learning workflows. The project showcases best practices for tracking machine learning experiments when multiple users collaborate on the same project.

## Architecture Diagram
![MLflow DagsHub Architecture](Architecture/DMML_Architecture.jpg)

### Key Objectives

1. **Experiment Tracking with MLFlow**: Log all model training runs with hyperparameters, metrics, and artifacts
2. **Centralized MLflow Backend**: Use DagsHub as a centralized MLflow tracking server for multi-user collaboration
3. **Model Versioning & Registry**: Register and manage models with MLflow Model Registry
4. **Reproducibility**: Ensure experiments are reproducible and comparable across different users and runs
5. **Evaluation & Artifact Management**: Log confusion matrices, ROC curves, and model artifacts for each experiment

---

## Folder Structure

```
DMML-Assignment/
│
├── README.md                          # Project overview and quick start guide
├── requirements.txt                   # Python dependencies
├── DOCUMENTATION.md                   # This file - Comprehensive project documentation
│
├── notebooks/                         # Jupyter Notebooks for different stages
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── model_training_and_mlflow_local.ipynb           # Training with local MLflow (debugging)
│   ├── model_training_and_mlflow_DAGSHUB.ipynb         # Training with DagsHub MLflow
│   ├── model_training_and_mlflow_DAGSHUB Shanmathy.ipynb  # Alternative user's training
│   └── model_load_predict.ipynb      # Load trained model and make predictions
│
├── data/                              # Dataset storage
│   └── processed/
│       └── heart_clean.csv           # Processed heart disease 
|
dataset (UCI ML Repo)
│
├── artifacts/                         # Generated evaluation plots
│   └── [model_name]_[metric].png     # Confusion matrices and ROC curves per run
│
├── models/                            # Saved model files (Local)
│   └── heart_disease_pred_model.pkl  # Best model saved as pickle
│
├── mlruns/                            # Local MLflow tracking 
│
└── dmmlenv/                           # Python virtual environment
```

---

## Project Workflow

### Phase 1: Exploratory Data Analysis (EDA)

**Notebook**: `eda.ipynb`

- Load and inspect the heart disease dataset
- Analyze feature distributions and correlations
- Identify categorical and numerical columns
- Handle missing values and prepare data

---

### Phase 2: Model Training with Experiment Tracking

#### Option A: Local MLflow Tracking

**Notebook**: `model_training_and_mlflow_local.ipynb`

Used for initial development and debugging:
- Logs experiments to local `mlruns/` directory
- Single-user workflow
- Fast iteration without external dependencies
- Useful for testing before pushing to DagsHub

#### Option B: DagsHub Centralized Tracking (Primary)

**Notebook**: `model_training_and_mlflow_DAGSHUB.ipynb`

The main collaborative workflow:
- Logs experiments to DagsHub's centralized MLflow server
- Supports multi-user collaboration
- All users see the same experiment history
- Enables comparison of different users' experiments
- Uses environment variables for configuration

---

## Experiment Tracking Details

### Tracked Artifacts Per Run

For each model training run, the following are logged:

#### 1. **Hyperparameters**
- Model type (Logistic Regression / Random Forest)
- Model-specific parameters (C, n_estimators, max_depth, etc.)
- Preprocessing settings (scaler type, encoder type)

#### 2. **Metrics**
- `test_accuracy`: Overall accuracy on test set
- `test_precision`: Precision score
- `test_recall`: Recall score
- `test_roc_auc`: ROC-AUC score

#### 3. **Artifacts**
- **Model**: Trained sklearn pipeline saved as MLflow model
- **Confusion Matrix**: PNG visualization showing true vs predicted labels
- **ROC Curve**: PNG visualization showing classification performance across thresholds

#### 4. **Tags**
- `model_type`: Type of model being trained
- `runner_name`: Name of the user/runner executing the experiment

### Models Evaluated

#### 1. Logistic Regression
```
Hyperparameter Grid:
- C: [0.1, 1, 10]
Total runs: 3
```

#### 2. Random Forest Classifier
```
Hyperparameter Grid:
- n_estimators: [100, 200]
- max_depth: [5, 10]
Total runs: 4
```

**Total training runs per experiment**: 7 different model configurations

---

## MLflow Integration

### MLflow Components Used

1. **Tracking Server**
   - Local: File-based storage in `mlruns/` directory
   - Remote: DagsHub's hosted MLflow instance

2. **Experiment Management**
   - Create experiments with timestamp-based names
   - Organize runs under experiments
   - Tag runs with metadata (model type, runner name)

3. **Model Registry**
   - Register best model to MLflow Model Registry
   - Version control for models
   - Support for different deployment stages (None/Staging/Production)

4. **Model Versioning**
   - Automatic versioning of registered models
   - Track model lineage and history
   - Enable rollback to previous versions

---

## DagsHub Integration (Multi-User Collaboration)

### Setup Requirements

Create a `.env` file in the project root:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-api-token>
RUNNER_NAME=<your-username>
```

### How DagsHub Enables Collaboration

1. **Centralized Backend**
   - All users push experiments to the same server
   - No need for manual file synchronization
   - Single source of truth for all experiment runs

2. **Shared Experiment History**
   - Users can view experiments from other users
   - Compare performance across different configurations
   - Track who ran which experiment (via `RUNNER_NAME` tag)

3. **Version Control Integration**
   - Integrates with Git/GitHub for code versioning
   - Links experiments to specific code commits
   - Enables reproducibility of results

4. **Web UI Access**
   - Dashboard for browsing experiments and runs
   - Compare metrics and parameters across runs
   - View logged artifacts (models, plots)

---

## Phase 3: Model Loading and Prediction

**Notebook**: `model_load_predict.ipynb`

Demonstrates production inference:

### Two Loading Strategies

1. **Load from MLflow Registry** (Recommended for production)
   ```python
   model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
   model = mlflow.sklearn.load_model(model_uri)
   ```
   - Uses registered model from MLflow Model Registry
   - Supports version specification
   - Enables smooth model updates without code changes

2. **Load from Local File** (Fallback)
   ```python
   model = joblib.load("../models/heart_disease_pred_model.pkl")
   ```
   - Loads previously saved pickle file
   - Useful if MLflow Registry is unavailable
   - Faster loading for local development

### Prediction Workflow

1. Load trained model (from MLflow or local)
2. Prepare input data (same preprocessing as training)
3. Generate predictions
4. Return predictions or probabilities

---

## Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | 2.2.3 | Data manipulation and analysis |
| **numpy** | 1.26.4 | Numerical computations |
| **scikit-learn** | 1.5.2 | Machine learning models and preprocessing |
| **joblib** | 1.4.2 | Model serialization and parallel processing |
| **mlflow** | 2.12.2 | Experiment tracking and model management |
| **matplotlib** | 3.10.3 | Visualization (EDA and evaluation plots) |
| **seaborn** | 0.13.2 | Statistical visualizations |
| **python-dotenv** | 1.0.1 | Environment variable management |
| **ucimlrepo** | - | UCI ML Repository dataset access |

### Environment

- **Python**: 3.11
- **Virtual Environment**: venv (in `dmmlenv/`)

---

## Dataset: Heart Disease

**Source**: UCI Machine Learning Repository

**Description**:
- Classification task: Predict presence of heart disease
- Features: Age, sex, cholesterol, blood pressure, etc.
- Target: Binary (disease present/absent)
- Preprocessing: Handled in EDA notebook
- Cleaned data location: `data/processed/heart_clean.csv`

---

## Key Features Demonstrated

### 1. **Hyperparameter Optimization**
- Grid search across multiple hyperparameter combinations
- Tracks all configurations and their performance
- Identifies best performing model automatically

### 2. **Evaluation Metrics Logging**
- Multiple metrics per run (accuracy, precision, recall, ROC-AUC)
- Enables comprehensive model evaluation
- Supports trade-off analysis (precision vs recall)

### 3. **Artifact Management**
- Log evaluation plots alongside models
- Store confusion matrices and ROC curves
- Enable visual inspection of model performance

### 4. **Model Comparison**
- MLflow UI allows side-by-side comparison of runs
- Filter and sort by metrics
- Identify best model across all experiments

### 5. **Reproducibility**
- All hyperparameters logged with metrics
- Random seeds set for reproducible splits
- Experiment names with timestamps prevent overwrites
- Code version tracked through Git integration with DagsHub

### 6. **Collaborative Features**
- `RUNNER_NAME` tag identifies who ran each experiment
- DagsHub provides pull-request-like workflow for experiments
- Comments and discussions on experiment runs
- Centralized model registry for all team members

---

## Running the Project

### 1. Setup Virtual Environment

```bash
# Activate the existing virtual environment
.\dmmlenv\Scripts\activate
```

### 2. Install Dependencies (if needed)

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` file:

```env
# For local MLflow (default)
# No variables needed - uses local mlruns/

# For DagsHub (remote)
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<api-token>
RUNNER_NAME=<your-name>
```

### 4. Run Notebooks in Order

1. **EDA.ipynb** - Explore data
2. **model_training_and_mlflow_DAGSHUB.ipynb** - Train and track experiments
3. **model_load_predict.ipynb** - Load model and make predictions

---

## Learning Outcomes

By completing this assignment, we will understand:

1. ✅ How to use MLflow for experiment tracking
2. ✅ How to log parameters, metrics, and artifacts
3. ✅ How to use MLflow Model Registry for model management
4. ✅ How DagsHub enables centralized, collaborative ML workflows
5. ✅ Best practices for reproducible ML experimentation
6. ✅ Multi-user collaboration in ML projects
7. ✅ Model versioning and production deployment readiness

---

## Advanced Features

### Experiment Naming Convention

Experiments are created with timestamp-based names:
```
heart_disease_experiment_runs_2024-12-15_143022
```

This prevents overwrites and creates a unique ID for each experiment batch.

### Run Naming Convention

Individual runs follow this pattern:
```
{model_name}_{hyperparameters}_{runner_name}

Example:
random_forest_n_estimators=100_max_depth=5_local_runner
```

### Best Model Selection

For each algorithm:
- All configurations are trained
- Best configuration tracked by accuracy
- Stored in `best_models` dictionary
- Overall best model selected across all algorithms

### Model Persistence

Best model saved in two formats:

1. **MLflow Model**
   - Location: MLflow registry
   - Format: Sklearn pipeline
   - Advantage: Version controlled, deployable

2. **Pickle File**
   - Location: `models/heart_disease_pred_model.pkl`
   - Format: Joblib pickle
   - Advantage: Quick local loading

---

## Troubleshooting

### DagsHub Connection Issues

Check:
- `.env` file contains correct URI
- URI format: `https://dagshub.com/username/repo.mlflow`
- API token has MLflow permissions
- Internet connection to dagshub.com

### Local MLflow Fallback

If DagsHub is unavailable:
- Set `MLFLOW_TRACKING_URI` to empty or missing
- Project automatically falls back to local `mlruns/`
- All functionality works locally

### Model Loading Failures

Ensure:
- Model was registered successfully
- MLflow tracking server is accessible
- Local pickle file exists as fallback
- Same preprocessing applied to input data

---

## References

- [MLflow Documentation](https://mlflow.org/docs)
- [DagsHub MLflow Integration](https://dagshub.com/docs/mlflow_integration)
- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

---
