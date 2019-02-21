import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import itertools
df = pd.read_csv('numerical.csv')

positives = np.zeros()
negatives = np.zeros()

for row in df['Customer Lifetime Value']:
    if row > df['Customer Lifetime Value'].mean():
        positives.append(1)
    else:
        negatives.append(1)

df['Positives'] = positives     
df['Negatives'] = negatives     
print(df['Positives'])
print(df['Negatives'])