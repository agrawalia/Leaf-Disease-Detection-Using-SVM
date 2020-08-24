# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import GaussianNB
from sklearn.naive_bayes import GaussianNB
import cv2
# import warnings to remove any type of future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# reading csv file and extracting class column to y.
dataf = pd.read_csv("Datasetinfectedhealthy.csv")

# extracting two features
X = dataf.drop(['imgid','fold num'], axis=1)
y = X['label']
X = X.drop('label', axis=1)

print("\nTraining dataset:-\n")
print(X)


log = pd.read_csv("D:/8semproject/Automatic-leaf-infection-identifier-master/Datasetunlabelledlog.csv")

log = log.tail(1)
X_ul = log.drop(['imgid','fold num'], axis=1)

print("\nTest dataset:-\n")
print(X_ul)

print("\n*To terminate press (q)*")

Sum = 0

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split  
for n in range(4):
	x_train, Xi_test, y_train, yi_test = train_test_split(X, y, test_size=0.25, random_state=60)  
	if cv2.waitKey(1) == ord('q' or 'Q'): break     
	svclassifier = svm.SVC(kernel='linear')  
	svclassifier.fit(x_train, y_train)  
	pred = svclassifier.predict(X_ul)
	
	Sum = Sum + pred
print("Accuracy: {}%".format(svclassifier.score(Xi_test, yi_test) * 100 ))


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(svclassifier, Xi_test, yi_test)                               
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)


print("\nprediction: %d" %int(Sum/4))

if(Sum < 2):
	print("The leaf is sufficiently healthy!")
else:
	print("The leaf is infected!")

print("\nKeypress on any image window to terminate")
cv2.waitKey(0)
