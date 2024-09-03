import streamlit as st
import numpy as np
from PIL import Image
from joblib import load
model = load("my_model.joblib")
# st.write(model)

def classify(image):
    img = Image.open(image)
    img = img.resize(size=(64,64))
    img = np.array(img)
    # print(img)
    # st.write(img.shape)
    a1 = np.expand_dims(img,axis=0)
    op = model.predict(a1)
    if  op == 0:
        st.write("The image is found to be CAT ")
    else :
        st.write("The image is found to be SATHISH ")



st.title("Welcome to Face detection ",)
st.write("please upload your file below\n")

x = st.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
if st.button("classify"):
    st.write("welcome to my show")
    classify(x)