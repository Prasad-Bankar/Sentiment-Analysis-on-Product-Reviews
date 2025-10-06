# Sentiment Analysis on Product Reviews

Short: A complete repo template to build a sentiment analysis pipeline (EDA → preprocessing → training → evaluation → small API).

## Structure
- `data/` : place raw datasets (e.g., `reviews.csv`)
- `notebooks/eda.ipynb` : exploratory analysis
- `train.py` : training script
- `predict.py` : inference script
- `utils.py` : helper functions
- `requirements.txt`

## How to run
1. Create venv and install: `pip install -r requirements.txt`
2. Place dataset in `data/reviews.csv` (columns: `text`,`rating` or `label`)
3. Train: `python train.py --data reviews.csv --model_dir models/`
4. Predict: `python predict.py --model models/model.pkl --text "I love this product"`

