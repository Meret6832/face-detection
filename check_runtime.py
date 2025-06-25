import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


df = pd.read_csv(sys.argv[1])

sns.relplot(data=df, x="n_pixels", y="t", hue="cat", style="res")
sns.regplot(data=df[df["correct"] == False], x="n_pixels", y="t", scatter=False, line_kws={'linewidth':1, 'linestyle': 'dashed', 'color': 'grey'}, label="regression res=False")
sns.regplot(data=df[df["correct"] == True], x="n_pixels", y="t", scatter=False, line_kws={'linewidth':1, 'linestyle': 'dotted', 'color': 'grey'}, label="regression res=True")

plt.show()