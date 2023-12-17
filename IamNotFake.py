#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np


# In[2]:


def load_and_preprocess_data(folder_path,label):
    images=[]
    labels=[]
    
    for filename in os.listdir(folder_path):
        img_path=os.path.join(folder_path,filename) 
        img=cv2.imread(img_path)
        img=cv2.resize(img,(64,64))
        img=img/255.0
        images.append(img)
        labels.append(label)
    return images,labels

real_images,real_labels=load_and_preprocess_data('dataset/Real',label=0)
fake_images,fake_labels=load_and_preprocess_data('dataset/Fake',label=1)

all_images=np.concatenate([real_images,fake_images],axis=0)
all_labels=np.concatenate([real_labels,fake_labels],axis=0)


# In[3]:


all_images


# In[4]:


all_labels


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)


# In[7]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load the pre-trained VGG16 model without the classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the pre-trained layers so they're not trainable
for layer in base_model.layers:
    layer.trainable = False

# Add your own classification layers on top of the pre-trained model
x = Flatten()(base_model.output) 
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


# In[8]:


# Save the model
model.save('my_image_classifier_model.h5')


# In[9]:


from tensorflow.keras.models import load_model

loaded_model = load_model('my_image_classifier_model.h5')

