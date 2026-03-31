import pandas as pd
from pprint import pprint as pp

df = pd.read_pickle("./fsapt_analysis_results.pkl")
print(df)
pp(df.columns.tolist())
for i, row in df.iterrows():
    pp(row)

