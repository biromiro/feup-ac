import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import os

mapping = {
    'A': 0,
    'B': 1,
    'C': 0,
    'D': 1,
}

mapping2 = {
    -1: 1,
    1: 0,
}
vals = pd.read_csv(
    f'{os.path.dirname(__file__)}/output/loan_full_dataset.csv', sep=";")
vals['status'] = vals['status'].map(lambda x: mapping[x])
vals.set_index('loan_id', inplace=True)
vals = vals[['status']]

dev = pd.read_csv(
    f'{os.path.dirname(__file__)}/data/loan_dev.csv', sep=";")
dev['status'] = dev['status'].map(lambda x: mapping2[x])
dev.set_index('loan_id', inplace=True)
dev = dev[['status']]

together1 = pd.merge(vals, dev, left_index=True, right_index=True)
if (len(together1.query('status_x != status_y')) != 0 and len(together1) != len(dev)):
    raise Exception("Dataset is not the same as the one used for training")

comp = pd.read_csv(
    f'{os.path.dirname(__file__)}/data/loan_comp.csv', sep=";")
comp.set_index('loan_id', inplace=True)
comp = comp[['status']]


def evaluate_result(df):
    together = pd.merge(df, vals, left_index=True, right_index=True)

    if (len(together) != len(comp)):
        raise Exception(
            f"Dataset is not the same as the predicted one: comp: {len(comp)} together: {len(together)} df:{len(df)}")

    predvals = together['Predicted'].values
    status = together['status'].values
    incomplete = roc_auc_score(status[:177], predvals[:177])
    real = roc_auc_score(status, predvals)
    print(
        f"You'd think it's something close to {incomplete} but it's actually {real}.")
    return real


def run(predfilename):
    df = pd.read_csv(
        f'{os.path.dirname(__file__)}/output/predictive/{predfilename}.csv')
    df.set_index('Id', inplace=True)
    return evaluate_result(df)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Running in interactive mode. Please press Ctrl+C to exit.")
        while True:
            print("Please enter the filename of the prediction you want to test. (relative to output/predict and without .csv)")
            run(input())
    else:
        run(sys.argv[1])
