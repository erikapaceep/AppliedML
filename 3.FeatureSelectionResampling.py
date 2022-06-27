import csv
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

np.set_printoptions(precision=3)

number_of_feature_to_include = 3
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(filename, names=names)
array = data.values
xx=(len(names))
#separate input and target where the target is always the last column
X = array[:,0:len(names)-1]
Y = array[:,len(names)-1]

#feature selection using Kbest
kbest = SelectKBest(score_func=chi2, k=number_of_feature_to_include)
kbest_fit = kbest.fit(X,Y)
print(kbest_fit.scores_)
features = kbest_fit.transform(X)
print(features[0:5,:])
print(kbest_fit.get_feature_names_out(names[0:len(names)-1]))
print(kbest_fit.get_support(indices=True))

Featureschi2 = list(zip(names[0:len(names)-1],kbest_fit.scores_))
print(Featureschi2)

#feature selection using RFE
estimator = LogisticRegression(solver="liblinear")
selector = RFE(estimator, n_features_to_select=5, step=1)
fit = selector.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
print("Feature names:", fit.get_feature_names_out(names[0:len(names)-1]))

#Feature extraction with PCA
pca = PCA(n_components=number_of_feature_to_include).fit(X)
print("explained variance: %s " % pca.explained_variance_ratio_)

#feature extraction with Extra tree classifier
model = ExtraTreesClassifier().fit(X,Y)
print(model.feature_importances_)
