import argparse, joblib
def main(args):
    model = joblib.load(args.model)
    text = args.text
    pred = model.predict([text])[0]
    print(f'Prediction: {pred}')

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--text', required=True)
    args=p.parse_args()
    main(args)
