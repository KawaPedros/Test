import numpy as np
import scipy.sparse

import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.externals.joblib import Memory

from sklearn.feature_extraction.text import TfidfVectorizer

fname_train = r"train.tsv"

fname_test = r"test_noy.tsv"

batch_size = 100000

def load_data(fname):
	X_load_data = []
	y_load_data = []
	with open(fname, newline='', encoding='utf-8') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			if row[0] == 'pants-fire':
				y_load_data.append(1)
			else:
				y_load_data.append(0)
			X_load_data.append(row[1])
	return X_load_data, y_load_data
	
if __name__ == "__main__":

	print("kawonczykm_lab2 started")

	X_train = []
	y_train = []

	X_test = []
	y_test = []
	
	X_raw = []
	y = []
	X_raw, y = load_data(fname_train)
	print("Train data loaded")
	#print(X_raw[0])
	#print(y[0])
		
	#ftv = TfidfVectorizer()
	#X_train = ftv.fit_transform(X_raw)
	#y_train = y
	
	#X_raw = []
	#X_raw, y = load_data(fname_test)
	#print("Test data loaded")
	#print(X_raw[0])
	#print(y[0])
	
	#X_test = ftv.transform(X_raw)
	#y_test = y
	
	train_size = 0.8
	test_size = 1. - train_size
	X_train, X_test, y_train, y_test = train_test_split(X_raw, y, train_size = train_size, test_size = test_size, random_state = 24)
	
	ftv = TfidfVectorizer()
	X_train = ftv.fit_transform(X_train)
	X_test = ftv.transform(X_test)
	
	print("LogisticRegression")
	lr = LogisticRegression()
	lr.fit(X_train, y_train)
	
	scores = lr.predict_proba(X_test)[:,1] # numerical score
	fpr, tpr, thresholds = roc_curve(y_test, scores)
	print(auc(fpr, tpr)) # area under the curve

	#with open("kawonczykm_output.res", "w") as text_file:
	#	text_file.write("Maciej Kawończyk\n")
	#	for item in rfc_scores:
	#		text_file.write("%s\n" % str(item))
