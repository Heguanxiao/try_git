import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score


data = pd.read_csv('train.csv')
X_tr = data.values[:,1:].astype(float)
y_tr = data.values[:,0]

neighbors = [5,7,10,13,15]
scores = list()
scores_std = list()

for neighbor in neighbors:
	neigh = KNeighborsClassifier(n_neighbors = neighbor)
	score = cross_val_score(neigh , X_tr,y_tr,cv=5)
	scores.append(np.mean(score))
	scores_std.append(np.std(score))

flag = 0;
for neighbor in neighbors:
	print 'neighbors: ', neighbor ,"; the scores: " , scores[flag] , "; the std: ",scores_std[flag]
	flag=flag+1

writer = open('predict.csv','w')
count = 1
writer.write('"ImageId","Label"\n')
for p in y_te:
	writer.write(str(count)+',"'+str(p)+'"\n')
	count = count +1
