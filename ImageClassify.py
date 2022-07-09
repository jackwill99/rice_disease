import keras
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import streamlit as st

# labels 
labels = [
    {
    "my": 'ရွက်ညိုပျောက်',
    "en": 'BrownSpot'
    },
    {
    "my": 'ပိုးကင်းစင်',
    "en": 'Healthy'
    },{
    "my": 'စပါးကျောက်ဆူး',
    "en": 'Hispa'
    },
    {
    "my": 'စပါးဂုတ်ကျိုး',
    "en": 'LeafBlast'
    },
    ]

model = keras.models.load_model("RiceDiseaseDetection60.h5")

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (200, 200)
    print('printing image ##########  ',image)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    result = np.argmax(prediction) # return position of the highest probability
    print("**********",result)
    return result


# start here
st.header("စပါးပိုးလေးတွေ ရှာကြည့်ရအောင်")

uploaded_file = st.file_uploader("ပုံလေးတွေ ထည့်ပေးပါ", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption='ပုံထည့်ပြီးပါပြီ', use_column_width=True)
    st.write("")
    st.write("ခွဲခြားနေသည်...")
    predict = teachable_machine_classification(image, 'RiceDiseaseDetection60.h5')

    print("############ ",labels[predict])
    # myanmar translate
    if (labels[int(predict)]["en"] =="Healthy"):
        st.success("ပိုးကင်းစင်ပါသည်")
    else:
        ending = f"{labels[int(predict)]['my']} ရောဂါပိုးဖြစ်ပါသည်"
        st.success(ending)

    link = f"[စပါးပိုးနဲ့ ပတ်သက်ပြီး အင်တာနက်တွင် ရှာကြည့်ရန်...](https://www.google.com/search?q=rice+disease+{labels[int(predict)]['en']})"
    st.markdown(link, unsafe_allow_html=True)