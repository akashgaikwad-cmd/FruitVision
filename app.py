import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
model=load_model('C:\python\imageClassification\Image_classify.keras')
data_categorys=['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_hight=180
img_width=180
image=st.text_input('Enter image','apple.jpg')

image_load=tf.keras.utils.load_img(image,target_size=(img_hight,img_width))
img_arr=tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict=model.predict(img_bat)

score=tf.nn.softmax(predict)
st.image(image)
st.write('Veg/Fruit in image is ' + data_categorys[np.argmax(score)])
st.write('With Accuracy of '+str(np.max(score)*100))