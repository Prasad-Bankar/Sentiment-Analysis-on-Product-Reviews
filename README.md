# Sentiment-Analysis-on-Product-Reviews

This project analyzes product reviews to predict customer sentiment (positive or negative).  
It includes steps for data preprocessing, feature extraction using TF-IDF, model training, and evaluation.

---

## 🧩 Project Structure
data_science_repos/sentiment-analysis/
├── data/ # place raw datasets (e.g., reviews.csv)
├── notebooks/ # EDA and experimentation
├── src/
│ ├── train.py # model training script
│ ├── predict.py # inference script
│ └── utils.py # helper functions
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Download dataset from Amazon Product Reviews – Kaggle
Prepare it as data/reviews.csv with columns:

arduino
Copy code
text, label
or map from ratings:

python
Copy code
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)
Train the model:

bash
Copy code
python src/train.py --data data/reviews.csv --model_dir models/
Test a prediction:

bash
Copy code
python src/predict.py --model models/model.pkl --text "I love this product!"
