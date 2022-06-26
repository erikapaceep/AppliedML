import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

#Univariate Histograms
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=names)
data.hist(figsize=(10,10),bins=20, grid=False)
pyplot.show()

#Univariate density plots
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,10), fontsize=5)
pyplot.show()

#Box and Whisker plots
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
data.plot(kind='box', subplots=True, layout=(3,3), sharey=False, figsize=(10,10))
pyplot.show()

#Correlation matrix plot
correlations = data.corr()
print(correlations)
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax=ax.matshow(correlations, vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

#scatterplot matrix
scatter_matrix(data, figsize=[15,15])
pyplot.show()

#Rescale data between 0 an 1 (useful for distance based methods)
from sklearn.preprocessing import MinMaxScaler
array = data.values
#separate inout from the target
X = array[:,0:8]
Y = array[:,8]
scaler = MinMaxScaler(feature_range=(0,1))
print(scaler)
rescaledX= scaler.fit_transform(X)
print(rescaledX[0:5,:])

#Standardise data (0 mean 1 stdev): it is more suitable for techniques that
#assume a Gaussian distribution in the input variables and work better with rescaled data
#such as linear and logistic regression and linear discriminant analysis
from sklearn.preprocessing import StandardScaler
import seaborn as sns
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

rescaledX_df = pd.DataFrame(rescaledX, columns=names[0:len(names)-1])
rescaledX_df.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,10), fontsize=5)
pyplot.show()
sns.distplot(rescaledX[:,1])
sns.distplot(rescaledX[:,2])
sns.distplot(rescaledX[:,3])
sns.distplot(rescaledX[:,4])
sns.distplot(rescaledX[:,5])
sns.distplot(rescaledX[:,6])
sns.distplot(rescaledX[:,7])

#Normalize data (rescale each observation, row, to have a length of 1 (unit norm)).
#this can be useful for sparse datasets with attributes of varying scaler
# (in distance based methods such as KNN)
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalizedX =scaler.transform(X)
print(normalizedX[0:5,:])

#binarizer
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
print(binaryX[0:5,:])

# Decision tree classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
model = DecisionTreeClassifier()
results1 = cross_val_score(model, X, Y, cv=kfold)
print("Mean estimated accuracy \n",results1.mean())

results2 = cross_val_score(model, normalizedX, Y, cv=kfold)
print("Mean estimated accuracy on normalized data \n",results2.mean())