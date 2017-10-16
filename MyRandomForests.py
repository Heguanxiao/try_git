import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


data = pd.read_csv('train.csv')
X_tr = data.values[:,1:].astype(float)
y_tr = data.values[:,0]

scores = list()
scores_std = list()

n_trees = [75,100,125,150,175]
for n_tree in n_trees:
	clf = RandomForestClassifier(n_estimators = n_tree)
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