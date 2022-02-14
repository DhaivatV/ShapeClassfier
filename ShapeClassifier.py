#!/usr/bin/env python
# coding: utf-8

# # IMPORT NECESSARY MODULES

# In[1]:


import cv2 as oc
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras. layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from joblib import dump, load


# # BUILD CLASSIFIER

# In[2]:


classifier = Sequential ()
classifier.add (Conv2D(32, (3,3), input_shape = (64,64,3), activation = "relu"))
classifier.add (MaxPooling2D(pool_size = (2,2)))
classifier.add (Conv2D(32, (3,3), activation = "relu"))
classifier.add (MaxPooling2D(pool_size = (2,2)))
classifier.add (Flatten())
classifier.add (Dense(units = 64, activation = "relu"))
classifier.add (Dense(units = 2, activation = "softmax"))
classifier.compile(optimizer = "adam",
           loss = "categorical_crossentropy", metrics = ["accuracy"]) 


# # GENERATE IMAGE DATA

# In[3]:


images = ImageDataGenerator (

rescale= 1./255, 
shear_range= 0.2,
zoom_range= 0.3,
horizontal_flip= True
    
)


# # SPLITTING DATA INTO TRAIN AND TEST DATA

# In[4]:


train_data = images.flow_from_directory(
   "toy_train/",
   target_size = (64,64),
   batch_size = 16,
   class_mode = "categorical")

test_data = images.flow_from_directory(
   "toy_val/",
   target_size = (64,64),
   batch_size = 16,
   class_mode = "categorical")
    


# # TRAINING

# In[5]:


classifier.fit(
train_data,
epochs = 10,
steps_per_epoch = len(train_data),
validation_data = test_data,
validation_steps = 20)
               
                


# # SAVING TRAINED MODEL

# In[6]:


classifier.save("model.h5")


# # TAKE  PREDICTIONS FROM MODEL

# In[7]:


img = image.load_img("toy_val/circle/118.jpg", target_size = (64,64))


# In[8]:


img_new  = image.img_to_array(img)


# In[9]:


img_new = np.expand_dims(img_new, axis = 0)


# In[10]:


print(train_data.class_indices)
prediction = classifier.predict(img_new)
print(prediction)


# In[ ]:




