import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA


data = pd.read_csv('train.csv')
X_tr = data.values[:,1:].astype(float)
y_tr = data.values[:,0]

pca = PCA(n_components=484)
pca.fit(X_tr)
X_tr = pca.fit_transform(X_tr)

scores = list()
scores_std = list()

neibors = [7,9,11]

for neibor in neibors:
	neigh = KNeighborsClassifier(n_neighbors=neibor)
	score = cross_val_score(neigh,X_tr,y_tr,cv=3)
	scores.append(np.mean(score))
	scores_std.append(np.std(score))

test = pd.read_csv('test.csv')
X_te = pca.fit_transform(test)





numbers = [125,150,175,200]
for number in numbers:
	clf = AdaBoostClassifier(n_estimators = number)
	score = cross_val_score(clf,X_tr,y_tr,cv=5)
	scores.append(np.mean(score))
	scores_std.append(np.std(score))

flag = 0;
for n_tree in n_trees:
	print 'Tree: ', n_tree ,"; the scores: " , scores[flag] , "; the std: ",scores_std[flag]
	flag=flag+1

clf.fit(X_tr,y_tr)
test = pd.read_csv('test.csv')
X_te = test.values[:,0:]
y_te = clf.predict(X_te)
writer = open('predict.csv','w')
count = 1
writer.write('"ImageId","Label"\n')
for p in y_te:
	writer.write(str(count)+',"'+str(p)+'"\n')
	count = count +1