import operator
import random

import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import streamlit.components.v1 as components

st.set_option('deprecation.showfileUploaderEncoding', False)


def classifier():
    new_model = tf.keras.models.load_model('modelLoadClassifier.h5')
    img = cv2.imread("temp.jpg")
    resized_image = cv2.resize(img, (30, 30))
    img = np.reshape(resized_image, (1, 30, 30, 3))
    predictions = new_model.predict(img)
    pred = np.array_str(predictions)
    temp2 = []
    dictValues = {}
    for num in pred.split():
        value = ''
        for char in num:
            if char != '[' and char != ']':
                value += char
        if value != '':
            temp2.append(float(value))
    for x in range(3):
        dictValues[x] = temp2[x]
    sorted_d = sorted(dictValues.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_d)
    classification = sorted_d[0][0]
    if classification == 0:
        print("compost")
        st.header("This item is Compost!")
        return "Compost"
    if classification == 1:
        print("recycle")
        st.header("This item is Recycle!")
        return "Recycle"
    if classification == 2:
        print("trash")
        st.header("This item is Trash!")
        return "Trash"


st.title('RecycleAI')
st.header('Sort your Waste')
st.subheader('RecycleAI uses AI technologies to help you decide should it be recycled, trashed, or composted.')
st.write("")

components.html(
    """<img class="recycle-img" src="https://image.flaticon.com/icons/svg/861/861143.svg" alt="recycle-img" 
    styles="height:250px; width:250px; padding:2rem;"> 
    """,
    height=250, width=250
)
st.header('Please upload an image file')
file = st.file_uploader("", type=["jpg", "png", 'jpeg'])

if file is None:
    st.text("")
else:
    image = Image.open(file)
    image.save("temp.jpg")
    st.image(image, width=300)
    classifier()

# streamlit run classifier.py

daily_challenges = []
daily_challenges.append('Go meatless for a day! Try to eat vegetarian or vegan foods!')
daily_challenges.append('Save water! Take an under 5 minute shower.')
daily_challenges.append('Help the Earth! Pick up 5 pieces of trash.')
daily_challenges.append('Clean transportation! Try to bike, walk, or carpool.')
daily_challenges.append('Cut paper waste! Reduce your use of paper. Use digital technologies.')
daily_challenges.append('Support your local businesses! Shop at local farmers market.')
daily_challenges.append('Go zero waste for one day! Try to use reusable products like water bottles and lunch boxes.')


st.header('Complete the Daily Challenge!')
st.write(random.choice(daily_challenges))

st.header('Environmental Resources:')
components.html("<a href='https://www.epa.gov/recycle/donating-food'>Donating Food</a>", height=50)
components.html("<a href='https://www.epa.gov/recycle/reducing-waste-what-you-can-do#Tips%20for%20Home'>What can you do</a>", height=50)
components.html("<a href='https://www.epa.gov/recycle/electronics-donation-and-recycling'>Donating Electronics</a>", height=50)