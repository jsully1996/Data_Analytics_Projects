from sympy.physics.quantum.circuitplot import matplotlib
from Cython.Shadow import inline
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv("numerical01.csv")

columns = dataset.columns.tolist()
columns = [c for c in columns if c not in ["Churn?","Customer ID","Age","clv","clv","resp","cov","edu","emp","incm","loc","mpa","twp","los","lr","gr","comm","mslc","mspi","noc","nop","nopp","pt","pr","ren","sales","tca","fb","count"]]
X = columns
Y = dataset['Churn?']
print(Y)