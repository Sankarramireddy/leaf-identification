import streamlit as st
from PIL import Image
from keras_preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model("BC.h6",compile=False)
lab = {0: 'Nerium Oleander', 1: 'Liriodendron Chinense', 2: 'Ginkgo Biloba', 3: 'Citrus Reticulata Blanco', 4: 'Cercis Chinensis', 5: 'Cedrus Deodara', 6: 'Acer Palmatum'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open("leaf.webp")
    img1 = img1.resize((224,224))
    st.image(img1,use_column_width=False)
    st.title("Leaf Identification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* LEAF DATA SET</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        

        if st.button("identify"):
            result = processed_img(save_image_path)
            st.success("identified leaf name  is: "+result)
run()
