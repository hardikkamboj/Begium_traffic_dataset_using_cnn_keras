# -*- coding: utf-8 -*-

import skimage
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from keras.models import Sequential
import keras
from sklearn.metrics import accuracy_score, confusion_matrix

#directory for train and test images folders
TRAIN_IMG_DIR = r'E:\programming\dataset\Belgium Traffic signs\Training'
TEST_IMG_DIR =  r'E:\programming\dataset\Belgium Traffic signs\Testing'

def load_data(data_dir):
    """
    input - dir of the folder which contains the data
    output - images, labels
    """
    #all the sub directories where each folder represents each label
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir,d))]
 
    #iterate over all the directories and store images and labels
    images = []
    labels = []
    
    for d in directories:
        label_dir = os.path.join(data_dir,d)
        file_names = [os.path.join(label_dir,f) for f in os.listdir(label_dir) if f.endswith('.ppm')]
        for file in file_names:
            images.append(skimage.data.imread(file))
            labels.append(int(d))
    return images,labels
    
images,labels = load_data(TRAIN_IMG_DIR)   

print('Unique classes: {0} \nTotal Images: {1}'.format(len(set(labels)), len(images)))

# Make a histogram with 62 bins of the `labels` data and show the plot: 
plt.figure(figsize = (10,8))
plt.hist(labels, 62)
plt.xlabel('Class')
plt.ylabel('Number of training examples')
plt.show()
     
    
# since all the images are not of the equal size, each image is resized
images32  = [skimage.transform.resize(img,(32,32), mode = 'constant') for img in images ]

images_array = np.array(images32)
labels_array = np.array(labels)

print(images_array.shape)
print(labels_array.shape)

print(labels[500:510])

#converting labels to one hot encodings
num_classes = 62
labels_oh = keras.utils.to_categorical(labels_array,62)

print(images_array.shape)
print(labels_oh.shape)


def display_images_and_labels(images,labels):
    unique_labels = set(labels)
    plt.figure(figsize = (15,15))
    i = 0
    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(8,8,i+1)
        plt.axis('off')
        plt.title('Label {0} ({1})'.format(label,labels.count(label)))
        plt.imshow(image)
        i += 1
    plt.show()
     
display_images_and_labels(images_array,labels)   

#shuffling the training set
images_array,labels_array = shuffle(images_array,labels_array)

print(labels_array[0:10])

def model():
    model = Sequential()
    model.add(Conv2D(filters = 128,kernel_size = (6,6),input_shape = (32,32,3),
                     activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(filters = 256,kernel_size = (4,4),activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units = 128,activation = 'relu'))
    model.add(Dense(units = num_classes,activation = 'softmax'))
    
    return model



#making a model
model = model()

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.fit(images_array,labels_oh,batch_size = 32,epochs = 3,verbose = 1)

model.save('model.h5')

#prediction for some random sample
random_indices = random.sample(range(images_array.shape[0]),10)
random_images = np.array([images_array[i] for i in random_indices])
random_labels = [labels_array[i] for i in random_indices]

random_preds = model.predict_classes(random_images)

random_images.shape

print(random_preds)
print(random_labels)


plt.figure(figsize = (10,10))
for i in range(10):
    plt.subplot(5,2,i+1)
    label = random_labels[i]
    pred = random_preds[i]
    plt.axis('off')
    color = 'green' if pred == label else 'red'    
    plt.text(40,10,'Truth is {}, pred is {}'.format(label,pred),fontsize = 12,color = color)
    plt.imshow(random_images[i])
    
    
test_images,test_y = load_data(TEST_IMG_DIR)    
   
test_images32  = [skimage.transform.resize(img,(32,32), mode = 'constant') for img in test_images ]

test_images_array = np.array(test_images32)    
predictions = model.predict_classes(test_images_array)

accuracy = accuracy_score(y_true = test_y,y_pred = predictions)   
print(accuracy)

cm = confusion_matrix(y_true  =test_y,y_pred = predictions)    
cm   



    
   
    
   
    
