import csv
import numpy as np
import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename,'rt')
reader = csv.reader(raw_data,delimiter=',')
x = list(reader)
data = np.array(x).astype('float')


# Load CSV using Pandas
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
peek = data.head(20)
print(peek)
types = data.dtypes
print(types)

# Statistical Summary
description = data.describe()
print(description)
class_counts = data.groupby('class').size()
print(class_counts)

# Pairwise Pearson correlations
correlations = data.corr(method = 'pearson')
print(correlations)

# Skew for each attribute
skew = data.skew()
print(skew)