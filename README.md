# Spotify Track Popularity Prediction
 
## Team Members
- Ani Kharabadze  
- Mariam Phirtskhalava  
- Gvantsa Tchuradze  
 
---
 
## Problem Statement
This project analyzes Spotify track data to predict track popularity based on audio features.  
The goal is to identify which musical characteristics contribute most to a track's popularity and develop machine learning models capable of predicting popularity categories (Low, Medium, High).
 
---
 
## Objectives
- Perform comprehensive exploratory data analysis on Spotify track features  
- Identify patterns and correlations between audio characteristics and popularity  
- Build and compare multiple machine learning classification models  
- Provide actionable insights for understanding track popularity factors  
 
---
 
## Dataset Description
**Source:** [Spotify Tracks Dataset - Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download)  
**Size:** 60,000 tracks (sampled from full dataset)
 
**Features:**  
- Audio characteristics: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo  
- Track metadata: track name, artist, album, genre, duration  
- Target variable: popularity (0-100), categorized as Low/Medium/High  
 
**Preprocessing:**  
- Missing values handled via median imputation (numerical) and 'Unknown' (categorical)
- Outliers capped using IQR method (1.5 × IQR) to preserve dataset integrity
- Data type optimization (bool, category, float32)
- Derived features created: `duration_min`, `popularity_label`, `energy_dance_ratio`, `is_instrumental`, `tempo_category`, `is_acoustic`, `valence_category`
 
---
 
## Installation and Setup
 
### Prerequisites
- Python 3.8 or higher  
- pip package manager  
 
### Installation Steps
```bash
# Clone the repository
git clone <repository-url>
cd spotify-track-analysis
 
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 
# Install dependencies
pip install -r requirements.txt
 
# Verify installation
python -c "import pandas; import sklearn; print('Installation successful')"
```
 
## Project Structure
```
spotify-track-analysis/
├── data/
│   ├── raw/                    # Original dataset (dataset.csv)
│   └── processed/              # Cleaned data (spotify_cleaned.csv)
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_eda_visualization.py
│   └── 04_machine_learning.py
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data cleaning pipeline
│   ├── eda.py                  # EDA functions and visualizations
│   └── models.py               # ML model implementations
├── reports/
│   ├── figures/                # EDA visualizations
│   └── results/                # Model outputs and metrics
├── models/                     # Saved models
├── README.md
├── CONTRIBUTIONS.md
├── requirements.txt
└── .gitignore
```
 
## Usage
### Data Processing
```bash
python src/data_processing.py
```
This will:
- Load raw data from `data/raw/dataset.csv`
- Handle missing values (categorical: 'Unknown', numerical: median)
- Cap outliers using IQR method (preserves dataset integrity)
- Convert data types (bool, category, float32 optimization)
- Create derived features: `duration_min`, `popularity_label`, `energy_dance_ratio`, `is_instrumental`, `tempo_category`, `is_acoustic`, `valence_category`
- Save cleaned data to `data/processed/spotify_cleaned.csv`
- Default: processes 60,000 samples
 
### Exploratory Data Analysis
```bash
python src/eda.py
```
Generates:
- Descriptive statistics with median
- Correlation heatmap (custom pink-green color scheme: #065F46, #B4E7CE, #FFB3D9, #D946A6)
- Distribution plots with mean/median lines for popularity, danceability, energy, valence
- Box plots for outlier detection across 7 audio features
- Categorical analysis: popularity distribution, tempo distribution, valence (mood) distribution, top 10 genres
- Scatter plots with polynomial trend lines: energy vs danceability, popularity vs energy, energy vs acousticness, danceability vs valence
- Violin plots: popularity by tempo category, energy by valence category
- Interactive HTML plot: duration vs tempo (plotly)
- Outlier analysis report with IQR method
- All saved
 
### Machine Learning
```bash
python src/models.py
```
Trains three classification models:
- **Logistic Regression** uses scaled features
- **Decision Tree** uses raw features
- **Random Forest** uses raw features
 
Outputs:
- Model performance metrics (accuracy, precision, recall, F1 score)
- Classification reports with per-class metrics
- 3 Confusion matrices
- 2 Feature importance plots for Decision Tree and Random Forest
- Model comparison chart (4-panel visualization showing accuracy, precision, recall, F1)
- Metrics summary CSV saved
 
**Feature Selection:**  
Models use 10 audio features: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_min
 
**Train-Test Split:** 80/20 with stratification on popularity_label
 
## Results Summary
### Model Performance
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 67.7%    | 67.1%     | 67.7%  | 66.9%    |
| Decision Tree       | 70.8%    | 71.5%     | 70.8%  | 70.4%    |
| Random Forest       | 75.4%    | 76.1%     | 75.4%  | 75.2%    |
 
**Best Model:** Random Forest with 75.4% accuracy and 75.2% F1 score
 
### Key Insights:
- Energy and Loudness show the strongest positive correlation (r = 0.72)
- Acousticness and Energy exhibit strong negative correlation (r = -0.65)
- Popularity shows weak correlation with individual audio features
- Tempo has minimal impact on popularity across all categories
- Valence (mood) distribution is skewed toward neutral and sad tracks
 
### Feature Importance:
Top 5 most important features for predicting popularity (from Random Forest):
1. Energy
2. Danceability
3. Loudness
4. Acousticness
5. Valence
 
## Methodology
### Data Preprocessing:
- **Missing Values**: Median imputation for numerical features, 'Unknown' for categorical features
- **Outlier Handling**: IQR method with capping
- **Data Type Optimization**: Converted to appropriate types
- **Feature Engineering**: Created 7 derived features: popularity categories, tempo categories, mood categories etc.
- **Excluded Columns**: id, popularity (raw), duration_ms, key, mode, time_signature, explicit (not used in outlier detection)
-Standardized features for model training
 
### Exploratory Data Analysis:
- Generated 9 visualization types across 8 saved files
- Performed correlation analysis on 10 audio features
- Conducted outlier analysis with IQR bounds reporting
- Analyzed categorical distributions across 4 derived features
- Created interactive HTML visualization for temporal relationships
 
### Machine Learning:
- Implemented three classification models with distinct configurations
- Used 80/20 train-test split with stratification on target variable
- Applied StandardScaler to features for Logistic Regression only
- Evaluated using accuracy, precision, recall, and F1 score (weighted average)
- Compared models using 4-panel visualization
- Selected best model based on F1 score ranking
 
## Limitations
- Dataset limited to 60,000 sampled tracks
- Dataset limited to acoustic genre tracks
- Popularity metric may be influenced by temporal factors not captured
- Model performance could improve with additional features (release date, artist popularity)
- External factors (marketing, playlisting) not included in analysis
 
## Future Work
- Incorporate temporal features (release date, trending status)
- Expand to multiple genres for broader applicability
- Implement deep learning models for improved accuracy
- Add real-time prediction API
- Integrate user listening history data
 
## Technical Details
**Languages:** Python 3.8+
 
**Libraries:**
- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn, plotly
- Machine Learning: scikit-learn
- Model Persistence: joblib
 
**Color Scheme:**
- Primary: #D946A6 (pink)
- Secondary: #065F46 (dark green)
- Accent: #B4E7CE (light green), #FFB3D9 (light pink)
 
**Development Tools:**
- Git for version control
- Virtual environment for dependency management
 
## References
- Dataset: [Spotify Tracks Dataset - Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download)
- Scikit-learn Documentation: https://scikit-learn.org
- Pandas Documentation: https://pandas.pydata.org
- Matplotlib Documentation: https://matplotlib.org
- Seaborn Documentation: https://seaborn.pydata.org
 
## License
This project is submitted as part of the Data Science with Python course.  
 
## Contact
- Gvantsa Tchuradze
- Mariam Phirtskhalava
- Ani Kharabadze
 
**Date:** January 2026  
**Course:** Data Science with Python  
**Institution:** Kutaisi International University
