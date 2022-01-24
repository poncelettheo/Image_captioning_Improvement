# Image_captioning_Improvement

## Objectives :
This work has been done for the course Computer exercises of Keio University. Its goal is to improve an already existing image captioning algorithm.
You can found the initial algorithm done by Ka Ho Tsang [here](https://github.com/kahotsang/image-captioning).

## Installation:
python==3.8.8

tensorflow==2.4.1 (or tensorflow-gpu==2.4.1)

nltk==3.6.5

## Structure
build_model : contains the tools to build the different models: initial, with Word2Vec, with Bidirectional LSTM ot with pretrained CNN.

Notebooks: the notebooks to build the models on Google Colab

sample_captioned:

![sample_1](https://github.com/poncelettheo/Image_captioning_Improvement/blob/main/sample_captioned/Figure_2.png)

![sample_2](https://github.com/poncelettheo/Image_captioning_Improvement/blob/main/sample_captioned/Figure_3.png)

![sample_3](https://github.com/poncelettheo/Image_captioning_Improvement/blob/main/sample_captioned/Figure_5.png)

caption_generator: to caption images of the sample_images folder using different models

