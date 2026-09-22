import pandas as pd
df = pd.read_excel(r"data\raw\NewsSumm\Processed\NewsSumm_processed.xlsx", nrows=0)
print(list(df.columns))
