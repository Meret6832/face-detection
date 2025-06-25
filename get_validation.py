import numpy as np
import os
import pandas as pd
import sys


def get_score(sub_df):
    false_pos = len(sub_df[(sub_df["res"] == True) & (sub_df["correct"] == False)])/len(sub_df)
    true_pos = len(sub_df[(sub_df["res"] == True) & (sub_df["correct"] == True)])/len(sub_df)
    false_neg = len(sub_df[(sub_df["res"] == False) & (sub_df["correct"] == False)])/len(sub_df)
    true_neg = len(sub_df[(sub_df["res"] == False) & (sub_df["correct"] == True)])/len(sub_df)

    print("n={} FP={:.2f} TP={:.2f} FN={:.2f} TN {:.2f}".format(len(sub_df), false_pos, true_pos, false_neg, true_neg))
    accuracy = (true_neg+true_pos)/(true_neg+false_pos+true_pos+false_neg)
    print("accuracy={:.2f}".format(accuracy), end=" ")
    try:
        precision = true_pos/(true_pos+false_pos)
    except ZeroDivisionError:
        precision = -1
    try:
        recall = true_pos/(true_pos+false_neg)
    except ZeroDivisionError:
        recall = -1
    if recall == -1 or precision == -1:
        f1 = -1
        print()
    else:
        f1 = 2*(precision*recall)/(precision+recall)
        print("precision={:.2f} recall={:.2f} f1={:.2f}".format(precision, recall, f1))
    print("run time: avg={:.2f} median={:.2f} max={:.2f}".format(np.average(sub_df.t), np.median(sub_df.t), np.max(sub_df.t)))


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])

    print("**OVERALL**")
    get_score(df)

    for folder in os.listdir("validation"):
        print(f"\n**{folder}**")
        sub_df = df[df["cat"] == folder]
        get_score(sub_df)