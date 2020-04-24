import pandas as pd


df = pd.read_csv("layer_outputs.txt", sep=',|\[|\t')
df =  df.T
print(df.head())
df.to_csv('layer_outputs.csv')