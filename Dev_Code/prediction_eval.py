from sklearn import svm
import boto3
from PIL import Image
import numpy as np
from StringIO import StringIO
from ast import literal_eval
from pylab import *
from sklearn import datasets, neighbors, linear_model
from sklearn.svm import SVC
import sklearn.neural_network 

pics = np.array(['th0_1.jpg' ,'th1_2.jpg' ,'th2_1.jpg' ,'th3_3.jpg' ,'th4_1.jpg' ,'th5_0.jpg' ,'th6_0.jpg' ,
                 'th7_2.jpg' ,'th8_3.jpg' ,'th9_3.jpg','th10_1.jpg','th11_2.jpg' ,'th12_3.jpg' ,'th13_1.jpg',
                 'th14_2.jpg','th15_0.jpg','th16_3.jpg','th17_2.jpg','th18_2.jpg','th19_2.jpg','th20_0.jpg',
                 'th21_0.jpg','th22_3.jpg','th23_1.jpg','th24_1.jpg','th25_3.jpg','th26_3.jpg','th27_2.jpg',
                 'th28_0.jpg','th29_1.jpg','th30_1.jpg','th31_2.jpg','th32_3.jpg','th33_0.jpg','th34_0.jpg',
                 'th35_1.jpg','th36_1.jpg','th37_1.jpg','th38_0.jpg','th39_0.jpg','th40_3.jpg','th41_3.jpg',
                 'th42_1.jpg','th43_1.jpg','th44_1.jpg','th45_0.jpg','th46_1.jpg','th47_1.jpg','th48_1.jpg',
                 'th49_0.jpg','th50_0.jpg','th51_0.jpg','th52_0.jpg','th53_1.jpg','th54_1.jpg','th55_2.jpg',
                 'th56_0.jpg','th57_0.jpg','th58_0.jpg','th59_0.jpg','th60_0.jpg','th61_0.jpg','th62_3.jpg',
                 'th63_2.jpg','th64_3.jpg','th65_2.jpg','th66_2.jpg','th67_2.jpg','th68_1.jpg','th69_1.jpg',
                 'th70_0.jpg','th71_0.jpg','th72_2.jpg','th73_2.jpg','th74_2.jpg','th75_2.jpg'])
n = len(pics)
im_array = np.zeros((n,76800))
#im_array = np.zeros((n,16384))
for i in range(n):
    im = np.array(Image.open("/Users/patrickkuiper/Desktop/Research/Sensors/Localization/Localization/Data/" 
                             + pics[i]).convert('L'))
    #im.resize((128,128))
    im_flat = im.flatten()
    im_array[i,:] = im_flat


labels = np.array([1,2,1,3,1,0,0,2,3,3,1,2,3,1,2,0,3,2,2,2,0,0,3,1,1,3,3,2,0,1,1,2,3,0,0,1,1,1,0,0,3,3,1,1,1,0,1,1,
                   1,0,0,0,0,1,1,2,0,0,0,0,0,0,3,2,3,2,2,2,1,1,0,0,2,2,2,2])

test_num = 10
train_number = n - test_num
X_train = im_array[:train_number]
Y_train = labels[:train_number]
X_test = im_array[train_number:]
Y_test = labels[train_number:]

clf = SVC()
clf.set_params(kernel='linear').fit(X_train, Y_train) 
print "SVM Prediction:"
print "True:", labels[train_number:]
print "Pred:", clf.predict(X_test)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, Y_train) 
print "KNN Prediction:"
print "True:", labels[train_number:]
print "Pred:", neigh.predict(X_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=4, max_depth=None, min_samples_split=2, random_state=0)
rf.fit(X_train, Y_train) 
print "Random Forest Prediction:"
print "True:", labels[train_number:]
print "Pred:", rf.predict(X_test)


from sklearn.ensemble import ExtraTreesClassifier
erf = ExtraTreesClassifier(n_estimators = 4, max_depth = None, min_samples_split = 2, random_state = 0)
erf.fit(X_train, Y_train) 
print "Extra Trees Prediction:"
print "True:", labels[train_number:]
print "Pred:", erf.predict(X_test)


from sklearn.tree import DecisionTreeClassifier
dt = ExtraTreesClassifier(n_estimators = 4, max_depth = None, min_samples_split = 2, random_state = 0)
dt.fit(X_train, Y_train) 
print "Decision Tree Prediction"
print "True:", labels[train_number:]
print "Pred:", dt.predict(X_test)


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train) 
print "Gaussian Naive Bayes Prediction:"
print "True:", labels[train_number:]
print "Pred:", nb.predict(X_test)

pic_num = 70
imshow(np.array(Image.open("/Users/patrickkuiper/Desktop/Research/Sensors/Localization/Localization/Data/" 
                           + pics[pic_num]).convert('L')))
show()