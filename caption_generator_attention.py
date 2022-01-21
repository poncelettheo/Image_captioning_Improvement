"""
This file contains code for sampling caption using the model built (default: sample_model.h5)
It will first convert all the images from the sample_images folder into a set of VGG16 features, 
and then pass the features to the trained deep-learning model and tokenizer,
and return the captions.
"""

import os
import keras
import keras.applications.vgg16
import numpy as np
import matplotlib.pyplot as plt
from build_model.build_model_transformers import feature_extractions, sample_caption
from pickle import load, dump
import gensim
    
#Load Word2Vec
word_model = gensim.models.Word2Vec.load("word2vec.model");
    
model = keras.models.load_model("./sample_model_attention.h5") #Load model
max_length = 33 #Maximum length of caption sequence

#sampling
features = feature_extractions("./sample_images")

for i, filename in enumerate(features.keys()):
    plt.figure(i+1)
    caption = sample_caption(model, max_length, features[filename])
    img = keras.preprocessing.image.load_img("./sample_images/{fn}.jpg".format(fn=filename))
    plt.imshow(img)
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()