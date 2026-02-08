📝 Automatic Essay Grader | Basic Essay Scoring System
https://img.shields.io/badge/Python-3.8+-blue?logo=python https://img.shields.io/badge/Model-BERT/TF--IDF-green https://img.shields.io/badge/License-MIT-yellow

<p align="center"> <img src="assets/plot.JPG" alt="Essay Score Distribution" width="300"/> </p>
A simple Python script that automatically grades essays using either BERT (if available) or TF-IDF with traditional machine learning models as a fallback.

✨ Features
✅ Dual Grading Approach - Tries BERT first, falls back to TF-IDF
✅ Basic Text Analysis - Word count, sentence count, lexical diversity
✅ Multiple ML Models - Linear Regression, Ridge, Random Forest
✅ Performance Metrics - MSE, MAE, R² scores
✅ Simple Visualizations - Score distribution and prediction plots

📋 Requirements
bash
pip install pandas numpy scikit-learn matplotlib seaborn
For BERT functionality (optional):

bash
pip install torch transformers
▶️ Usage
Prepare your dataset: Create a CSV file with columns full_text and score

Update the file path in line 14 of the script:

python
df = pd.read_csv('your_dataset.csv')  # Change this path
Run the script:

bash
python "Automatic Essay Grading using BERT.py"
📊 Project Structure
text
auto-essay-grader/
├── Automatic Essay Grading using BERT.py    # Main script
├── assets/                                  # Screenshots
│   ├── code.JPG                            # Code preview
│   ├── output.JPG                          # Program output
│   └── plot.JPG                            # Score distribution
└── README.md                               # This file
🖼️ Screenshots
Code	Output
https://assets/code.JPG	https://assets/output.JPG
Visualization
https://assets/plot.JPG
⚙️ How It Works
1. Data Loading
Loads essay data from CSV file

Selects relevant columns (full_text and score)

Cleans and preprocesses the data

2. Model Selection
BERT Approach (if torch/transformers installed):

Uses pre-trained BERT model

Fine-tunes for essay grading

TF-IDF Fallback (default):

Extracts text features (word count, sentence count, etc.)

Applies TF-IDF vectorization

Trains multiple regression models

3. Evaluation
Calculates MSE, MAE, and R² scores

Generates prediction vs actual plots

Displays training progress

📈 Sample Output
text
Essay Grading DataSet:
   essay_id  score  ...  source_text_4
0  AAAVUP...      4  ...            NaN

Score Statistics:
Minimum score: 1
Maximum score: 6
Average score: 2.94
Standard deviation: 1.04

Model Performance:
Linear Regression - R²: 0.72, MAE: 0.45
Random Forest - R²: 0.75, MAE: 0.42
🔧 Customization
Change Scoring Range
The dataset expects scores on a scale (typically 1-6). To adjust:

python
# Clip predictions to your score range
min_score, max_score = 1, 6
predicted_score = max(min_score, min(max_score, predicted_score))
Add New Features
Extend the feature extraction in the TF-IDF section:

python
# Add more text features as needed
features.append({
    'word_count': word_count,
    'sentence_count': sentence_count,
    # Add your features here
})
⚠️ Notes
Dataset Required: You need to provide your own ASAP.csv or similar dataset

Memory Usage: BERT requires significant RAM (8GB+ recommended)

First Run: The script will download BERT model files (~400MB) if using BERT approach

📝 License
MIT License - feel free to use and modify for your needs.
