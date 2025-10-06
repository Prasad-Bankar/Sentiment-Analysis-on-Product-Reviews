import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    if 'label' in df.columns:
        return df['text'].astype(str), df['label']
    # try mapping rating to binary
    if 'rating' in df.columns:
        y = df['rating'].apply(lambda r: 1 if float(r)>=4 else 0)
        return df['text'].astype(str), y
    raise ValueError('Expect columns text+label or text+rating')

def main(args):
    X, y = load_data(args.data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_val, y_val)
    print(f'Validation accuracy: {acc:.4f}')
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(args.model_dir, 'model.pkl'))
    print('Model saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--model_dir', default='models')
    args = parser.parse_args()
    main(args)
