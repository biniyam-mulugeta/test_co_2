#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/biniyam-mulugeta/test_co_2/blob/main/c06_with_ROC.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import tensorflow as tf
import numpy as np 
from keras.models import Model
from keras.layers import Flatten,concatenate#,Dense,Dropout,Conv2D
from keras.applications import densenet, mobilenet_v2#efficientnet#densenetmobilenet_v2#,efficientnet,vgg16,densenet
#from keras_preprocessing import image
#from keras_preprocessing.image import ImageDataGenerator
#from keras.utils.all_utils import to_categorical
from sklearn import preprocessing
#from skimage.filters import threshold_otsu
import cv2
from pathlib import Path
import os
import glob
import matplotlib.pyplot as plt
#import seaborn as sns
#import zipfile
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import classification_report,f1_score,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from skimage import io
from yellowbrick.classifier import ROCAUC


# In[4]:


def k_means_segmentation(img):
  twoDimage = img.reshape((-1,3))
  twoDimage = np.float32(twoDimage)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 2
  attempts=1
  ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
  center = np.uint8(center)
  res = center[label.flatten()]
  result_image = res.reshape((img.shape))
  return result_imag

def color_mask_segmentation(img):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  light_blue = (90, 70, 50)
  #dark_blue = (128, 255, 255)
  # You can use the following values for green
  light_green = (40, 40, 40)
  # dark_greek = (70, 255, 255)
  mask = cv2.inRange(hsv_img, light_blue, light_green)
  result = cv2.bitwise_and(img, img, mask=mask)
  return result

def filter_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])
  
def otsu_segmentation(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(img_gray)
    img_otsu  = img_gray < thresh
    filtered = filter_image(img, img_otsu)
    return filtered


# In[5]:


SIZE = 224
train_image = []
train_label = []
for dir_path in glob.glob("D:/bini/masters research/Dataset/coffee dataset/BRACOL_coffee_leaf_ images_datasets_G/dataset_extracted/train/train_cro/*"):
    label = dir_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(dir_path,"*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(SIZE,SIZE))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        #img = otsu_segmentation(img)
        train_image.append(img)
        train_label.append(label)
train_image = np.array(train_image)
train_label = np.array(train_label)

val_image = []
val_label = []
for dir_path_v in glob.glob("D:/bini/masters research/Dataset/coffee dataset/BRACOL_coffee_leaf_ images_datasets_G/dataset_extracted/validation/validation_cro/*"):
    label_v = dir_path_v.split("\\")[-1]
    #print(label_v)
    for img_path_v in glob.glob(os.path.join(dir_path_v,"*.jpg")):
        #print(img_path_v)
        img = cv2.imread(img_path_v,cv2.IMREAD_COLOR)
        img = cv2.resize(img,(SIZE,SIZE))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        #img = otsu_segmentation(img)
        val_image.append(img)
        val_label.append(label_v)
val_image = np.array(val_image)
val_label = np.array(val_label)


# In[6]:


img = train_image[123]
print(img.shape)
#img = cv2.imread(val_image[0])
io.imshow(img)
plt.show()


# In[7]:


le = preprocessing.LabelEncoder()
le.fit(train_label)
train_labe_encoded = le.transform(train_label)
le.fit(val_label)
val_label_encoded = le.transform(val_label)


# In[8]:


x_train,y_train,x_test,y_test = train_image,train_labe_encoded,val_image,val_label_encoded


# In[9]:


x_train,x_test = x_train/255.0,x_test/255.0


# In[10]:


#y_train_one_hot = to_categorical(y_train)
#y_test_one_hot = to_categorical(y_test)


# In[11]:


model_1 = densenet.DenseNet121()
#model.summary()


# In[12]:


model_2 = mobilenet_v2.MobileNetV2()


# In[13]:


#model.summary()


# In[14]:


out1 = model_1.get_layer(index = -2).output
out2 = model_2.get_layer(index = -2).output
out = concatenate([out1,out2])


# In[15]:


e_model = Model(inputs = [model_1.input,model_2.input],outputs = out)


# In[16]:


print("Feature extraction ...")
prediction = np.array(e_model.predict([x_train,x_train]))
prediction.shape


# In[17]:


Xtrain = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))


# In[18]:


prediction = np.array(e_model.predict([x_test,x_test]))


# In[19]:


Xtest = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))


# In[20]:


print('\tFeatures training shape: ', Xtrain.shape)
print('\tFeatures testing shape: ', Xtest.shape)


# In[21]:


print("Classification with Linear SVM ...")
svm = SVC(kernel='linear',probability=True)
svm.fit(Xtrain, np.ravel(y_train, order='C'))
result = svm.predict(Xtest)

acc = accuracy_score(result, np.ravel(y_test, order='C'))
print("\tAccuracy Linear SVM: %0.4f" % acc)


# In[22]:


skfolds = StratifiedKFold(n_splits=3,shuffle=True, random_state=42)
kf = KFold(n_splits=3)


# In[ ]:


score = []
i = 0
for trainset,testset in skfolds.split(Xtrain,y_train):
    print("folds ", i)
    print(trainset,"having :" , len(trainset))
    print(testset,"having :" , len(testset))

    x_tr,x_te=Xtrain[trainset],Xtrain[testset]
    y_tr,y_te=y_train[trainset],y_train[testset]

    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(x_tr, y_tr)

    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
    
    #svm.fit(x_tr,np.ravel(y_tr, order='C')) 
    #result_svm = svm.predict(x_te)
   # score.append(result_svm)
    #acc = accuracy_score(score[i], np.ravel(y_te, order='C'))
    #print("\tAccuracy random forest: %0.4f" % acc)
    #cm = confusion_matrix(y_te, result_svm, labels=svm.classes_)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svm.classes_)
    #disp.plot(cmap='Blues')
    #plt.show()
    #print("cross validation score :",cross_val_score(svm, x_tr, y_tr, cv=3, scoring="accuracy"))
    #print("F1 score is : ",f1_score(y_te, result_svm,average='micro'))
    print("iteration", i)
    i+=1


# In[33]:


score = []
i = 0
for trainset,testset in skfolds.split(Xtrain,y_train):
    print("folds ", i)
    print(trainset,"having :" , len(trainset))
    print(testset,"having :" , len(testset))

    x_tr,x_te=Xtrain[trainset],Xtrain[testset]
    y_tr,y_te=y_train[trainset],y_train[testset]

    rf.fit(x_tr,np.ravel(y_tr, order='C')) 
    result_rf = rf.predict(x_te)
    score.append(result_rf)
    acc = accuracy_score(score[i], np.ravel(y_te, order='C'))
    print("\tAccuracy Linear SVM: %0.4f" % acc)
    cm = confusion_matrix(y_te, result_rf, labels=svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svm.classes_)
    disp.plot(cmap='Blues')
    plt.show()
    print("iteration", i)
    i+=1


# In[46]:


accuracy = np.mean(score)
print(accuracy)


# In[32]:


acc = accuracy_score(score[2], np.ravel(y_te, order='C'))
print("\tAccuracy Linear SVM: %0.4f" % acc)


# In[18]:


cm = confusion_matrix(y_test, result, labels=svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=svm.classes_)
disp.plot(cmap='Blues')
plt.show()


# In[19]:


print(classification_report(y_test,result))


# In[31]:


rf = RandomForestClassifier(n_estimators=50,random_state=42)
rf.fit(Xtrain, np.ravel(y_train, order='C'))


# In[20]:


result_rf = rf.predict(Xtest)


# In[21]:


acc = accuracy_score(result_rf, np.ravel(y_test, order='C'))
print("\tAccuracy Linear SVM: %0.4f" % acc)
print(confusion_matrix(y_test,result))
print(classification_report(y_test,result))


# In[22]:


gb = GaussianNB()


# In[23]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[66]:


voting_clf = VotingClassifier(estimators=[('svc', svm), ('rf', rf),('gnb', gb),('knn',knn)],voting='soft')
voting_clf.fit(Xtrain, y_train)


# In[68]:


for clf in (svm,rf,gb,knn, voting_clf):
    clf.fit(Xtrain, y_train)
    y_pred = clf.predict(Xtest)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# ### ada boost ensemble classifier

# In[69]:


clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, Xtrain, y_train, cv=5)
scores.mean()


# ### calibration curve

# In[52]:


cm = confusion_matrix(y_test, result, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=rf.classes_)
disp.plot(cmap='BuGn')
plt.show()


# In[35]:


def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'healthy', 
                                        1: 'miner', 
                                        2: 'phoma',
                                        3: 'rust',
                                        4: 'molds'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer


# In[36]:


plot_ROC_curve(svm,Xtrain,y_train,Xtest,y_test)


# In[37]:


plot_ROC_curve(rf,Xtrain,y_train,Xtest,y_test)


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


# Utility function to move the midpoint of a colormap to be around
# the values of interest.


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# #############################################################################
# Load and prepare data set
#
# dataset for grid search


iris = load_iris()
X = iris.data
y = iris.target

# Dataset for decision function visualization: we only keep the first two
# features in X and sub-sample the dataset to keep only 2 classes and
# make it a binary classification problem.

X_2d = X[:, :2]
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

# It is usually a good idea to scale the data for SVM training.
# We are cheating a bit in this example in scaling all of the data,
# instead of fitting the transformation on the training set and
# just applying it on the test set.

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))

# #############################################################################
# Visualization
#
# draw visualization of parameter effects

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis("tight")

scores = grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(
    scores,
    interpolation="nearest",
    cmap=plt.cm.hot,
    norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
)
plt.xlabel("gamma")
plt.ylabel("C")
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title("Validation accuracy")
plt.show()


# In[ ]:




