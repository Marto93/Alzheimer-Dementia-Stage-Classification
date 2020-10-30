# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:25:10 2020

@author: Dr. Luca Martorano
"""
################## ALZHEIMER DEMENTIA STAGE PREDICTION ########################

#%% IMPORTING LIBRARIES

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image
import matplotlib.pyplot as plt

#%% IMPORTING DATA

# Defining the working directory

path = 'C:\\Users\\marto\\Desktop\\OASIS'
os.chdir('C:\\Users\\marto\\Desktop\\OASIS')

def importing(path):
    sample = []
    for filename in glob.glob(path):
        img = Image.open(filename, 'r')
        IMG = np.array(img)
        sample.append(IMG)
    return sample

path1 = 'C:\\Users\\marto\\Desktop\\OASIS\\train\\NonDemented\\*.jpg' 
path2 = 'C:\\Users\\marto\\Desktop\\OASIS\\train\\VeryMildDemented\\*.jpg'
path3 = 'C:\\Users\\marto\\Desktop\\OASIS\\train\\MildDemented\\*.jpg'
path4 = 'C:\\Users\\marto\\Desktop\\OASIS\\train\\ModerateDemented\\*.jpg'

train_ND = importing(path1)
train_VMD = importing(path2)
train_MID = importing(path3)
train_MOD = importing(path4)

#%% CREATION OF DATASETS

df_train_ND = pd.DataFrame({'image':train_ND, 'label': 'ND'})
df_train_VMD = pd.DataFrame({'image':train_VMD, 'label': 'VMD'})
df_train_MID = pd.DataFrame({'image':train_MID, 'label': 'MID'})
df_train_MOD = pd.DataFrame({'image':train_MOD, 'label': 'MOD'})

final_data = [df_train_ND, df_train_VMD, df_train_MID, df_train_MOD]
final_data = pd.concat(final_data)


#%% TRAIN LABEL SEPARATION

train_data = final_data['image']
labels = final_data['label']

#%% DATA NORMALIZATION

from sklearn.preprocessing import MinMaxScaler

def normalization(array):
    
    train_norm = []
    transformer = MinMaxScaler()
    
    for value in array:
        value = transformer.fit_transform(value)
        train_norm.append(value)
    
    return train_norm

train_norm = normalization(train_data)


#%% ENCODING THE LABELS

from sklearn.preprocessing import LabelBinarizer

onehot = LabelBinarizer()
labels = onehot.fit_transform(labels)

#%% TRAIN & TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_norm, labels,
                                                  test_size = 0.2,
                                                  stratify = labels,
                                                  shuffle = True,
                                                  random_state = 42)

X_train = np.array(X_train).reshape(5120,208,176,1)
X_test = np.array(X_test).reshape(1280,208,176,1)

#%% BALANCING THE DATA DURING TRAIN

from sklearn.utils import compute_class_weight

y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

#%% CREATING THE 'BASELINE' CNN MODEL 

import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Conv2D , MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import RMSprop

def build_model():
    
    '''Sequential Model creation'''
    Cnn = Sequential()
    
    Cnn.add(Conv2D(64,(5,5), activation = 'relu', padding = 'same',
                   strides=(2,2), input_shape = [208,176,1]))
    Cnn.add(MaxPooling2D(2))
    Cnn.add(Conv2D(128,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
    Cnn.add(Conv2D(128,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
    Cnn.add(Conv2D(256,(5,5), activation = 'relu', padding = 'same', strides=(2,2)))
    Cnn.add(MaxPooling2D(2))
    Cnn.add(Flatten())
    Cnn.add(Dense(64, activation = 'relu'))
    #Cnn.add(BatchNormalization())
    Cnn.add(Dropout(0.4))
    Cnn.add(Dense(32, activation = 'relu'))
    Cnn.add(Dropout(0.4))
    Cnn.add(Dense(4, activation = 'softmax'))
    
    '''Compiling the model'''
    Cnn.compile(optimizer = RMSprop(learning_rate = 1e-4), loss='categorical_crossentropy', 
              metrics =[ 'acc']) 
    
    return Cnn

keras_model = build_model()
keras_model.summary()

#%% FITTING THE MODEL

checkpoint_cb = ModelCheckpoint("AD_Stages_model.h5", save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10 ,
                   restore_best_weights=True, verbose=1)

history = keras_model.fit(X_train, y_train, validation_split = 0.1,
                    epochs= 100, batch_size = 10,# class_weight = d_class_weights ,
                    callbacks=[es, checkpoint_cb], verbose = 1)

keras_model.save('AD_Stages_model.h5')

#%% LOADING THE MODEL

keras_model = keras.models.load_model('AD_Stages_model.h5')
keras_model.compile(optimizer = RMSprop(learning_rate = 1e-4),
                    loss='categorical_crossentropy', metrics =[ 'acc'])
 
#%% PLOTTING RESULTS (Train vs Validation)

acc_history = history.history['acc']
val_acc_history = history.history['val_acc']

loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1,2, figsize= (20,15))
fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

ax1.plot(range(1, len(acc_history) + 1), acc_history)
ax1.plot(range(1, len(val_acc_history) + 1), val_acc_history)
ax1.set_title('History of Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(['training', 'validation'])


ax2.plot(range(1, len(loss_history) + 1), loss_history)
ax2.plot(range(1, len(val_loss_history) + 1), val_loss_history)
ax2.set_title('History of Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(['training', 'validation'])


#%% PREDICTION 

# Prediction on test_set

pred_test = keras_model.predict(X_test, verbose = 1)
pred_test = onehot.inverse_transform(pred_test)
real_val = onehot.inverse_transform(y_test)
pred_test_prb= keras_model.predict_proba(X_test)


#%% CONFUSION MATRIX

from collections import Counter
from sklearn.metrics import confusion_matrix

print(Counter(real_val))
print(Counter(pred_test))

conf_mx = confusion_matrix(real_val, pred_test)
conf_mx
 
heat_cm = pd.DataFrame(conf_mx, columns=np.unique(real_val), index = np.unique(real_val))
heat_cm.index.name = 'Actual'
heat_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(heat_cm, cmap="Blues", annot=True, annot_kws={"size": 16},fmt='g')# font size


#%% PLOTTING THE ROC & PRECISION-RECALL CURVES

from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve

fig, (ax1, ax2) = plt.subplots(1,2, figsize= (20,15))
fig.suptitle(" ROC AND P&R VISUALIZATION ")

# ROC Curve

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],pred_test_prb[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax1.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax1.set_title("ROC Curve")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="best")

# Precision-Recall Curve

precision = dict()
recall = dict()

for i in range(4):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], pred_test_prb[:, i])
    ax2.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    ax2.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')

ax2.set_title("PRECISION vs. RECALL Curve")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="best")

#%% LOOKING AT THE METRIC REPORT

from sklearn.metrics import classification_report

print(classification_report(real_val, pred_test))
print(roc_auc)


######################### FEATURE MAPS' VISUALIZATIONS ########################

#%% IMPORTING NECESSARY LIBRARIES

import keras
from matplotlib import pyplot
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras.models import Model
from skimage import color
from skimage import io

new_model = keras.models.load_model('AD_Stages_model.h5')
new_model.summary()

#%% SUMMARIZE FILTER SHAPES

# Summarize filters in each convolutional layer
from matplotlib import pyplot

for layer in new_model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# retrieve weights from the second hidden layer
filters, biases = new_model.layers[0].get_weights()

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()

###############################################################################


# summarize feature map shapes
for i in range(len(new_model.layers)):
	layer = new_model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
    
    
# redefine model to output right after the first hidden layer
model2 = Model(inputs=new_model.inputs, outputs=new_model.layers[3].output)

# load the image with the required shape
#img = load_img('C:\\Users\\marto\\Desktop\\OASIS\\train\\ModerateDemented\\27 (2).jpg', target_size=(208, 176))
img = color.rgb2gray(io.imread('C:\\Users\\marto\\Desktop\\OASIS\\train\\ModerateDemented\\27 (2).jpg'))

# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)

ixs = [0,2,3,4]
outputs = [new_model.layers[i].output for i in ixs]
model3 = Model(inputs=new_model.inputs, outputs=outputs)
feature_maps = model3.predict(img)

square = 2
for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()