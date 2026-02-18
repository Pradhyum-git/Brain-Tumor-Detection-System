import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
#st.title('Brain Tumor Detection System')
tumor_info = {
    "Glioma tumor": {
        "info": "Gliomas originate from glial cells and often show irregular patterns in MRI scans.",
        "precautions": [
            "Regular follow-up imaging is commonly recommended",
            "Maintain proper sleep and stress management",
            "Avoid head injuries",
            "Consult a specialist for persistent symptoms"
        ]
    },
    "Meningioma tumor": {
        "info": "Meningiomas arise from the protective membranes of the brain and are often slow-growing.",
        "precautions": [
            "Periodic imaging follow-ups",
            "Avoid physical trauma to the head",
            "Maintain a balanced lifestyle",
            "Monitor headaches or vision changes"
        ]
    },
    "Pituitary tumor": {
        "info": "Pituitary tumors occur in the pituitary gland and may affect hormonal balance.",
        "precautions": [
            "Regular clinical evaluations are important",
            "Maintain consistent sleep and diet routines",
            "Monitor unusual fatigue or vision issues",
            "Avoid self-medication"
        ]
    },
    "No Tumor": {
        "info": "No tumor-like abnormal patterns were detected by the model.",
        "precautions": [
            "Routine health checkups",
            "Seek medical advice if symptoms persist",
            "Maintain healthy lifestyle habits"
        ]
    }
}

class_labels =['No tumor','Pituitary','Glioma','Meningioma']
model = load_model('/home/rgukt/Desktop/ML PROJECTS/Medical-Image-Analysis-Brain-Tumor-Detection-main/vgg_base_model.h5')

def preprocess_test_image(image, img_size=224):
    image = image.resize((img_size,img_size))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)   # (1, H, W, C)
    image = preprocess_input(image)
    return image

   


# Sidebar
st.sidebar.title("Brain Tumor Detection Support")
page = st.sidebar.radio(
    "Go to",
    ["Prediction", "Confidence Comparison"]
)

# ---- PAGE 1 ----
if page == "Prediction":
    st.title("Prediction Page")
    img_file = st.file_uploader(label='choose a file',type=['jpg','png','jpeg'])
    
    if img_file is not None:
        img = Image.open(img_file)
        st.image(img,caption='uploaded image',use_container_width=True)
        tumor = None
        if st.button('Predict'):
            test_img = preprocess_test_image(img)
            pred = model.predict(test_img)[0]
            # st.write(pred)
            pred_class = np.argmax(pred)
            if(pred_class == 0):
                st.header('No Tumor')
                tumor ='No Tumor'
                confidence = pred[pred_class]
                st.write(f'Confidence : {confidence}')
            
            else:
                tumor = f'{class_labels[pred_class]} tumor'# tumor type
                confidence = pred[pred_class] # how much confidence that is that type of tumor is 
                st.header(tumor)
                st.write(f'Confidence : {confidence}')

            if confidence >= 0.85:
                st.success("Prediction Stability: High")
            elif confidence >= 0.65:
                st.warning("Prediction Stability: Moderate")
            else:
                st.error("Prediction Stability: Low")

    
        if tumor and tumor in tumor_info:
            st.markdown("### 🧠 Tumor Information & General Guidance")
            st.write("**General Information:**")
            st.write(tumor_info[tumor]["info"])

            st.write("**General Precautions (Non-Medical):**")
            for p in tumor_info[tumor]["precautions"]:
                st.write(f"- {p}")

            st.info(
                "Disclaimer: This information is educational only and not a medical diagnosis or treatment advice."
            )


# ---- PAGE 2 ----
elif page == "Confidence Comparison":
    st.title("📊 Prediction Confidence Comparison (Follow-up Analysis)")
    col1, col2 = st.columns(2)
    tumors = []
    confidences = []
    def confidence_trend(prev_conf, curr_conf):
        if curr_conf > prev_conf:
            return "Increase in prediction confidence observed"
        elif curr_conf < prev_conf:
            return "Decrease in prediction confidence observed"
        else:
            return "No significant change in prediction confidence"

    with col1:
        st.subheader("Previous Scan")
        prev_img_file = st.file_uploader(label='Upload Previous MRI',key='prev')
        # model = load_model('/home/rgukt/Desktop/ML PROJECTS/Medical-Image-Analysis-Brain-Tumor-Detection-main/vgg_base_model.h5')
        if prev_img_file is not None:
            prev_img = Image.open(prev_img_file)
            st.image(prev_img,caption='uploaded image',use_container_width=True)
                
            
            test_img = preprocess_test_image(prev_img)
            pred = model.predict(test_img)[0]
            # st.write(pred)
            pred_class = np.argmax(pred)
            if(pred_class == 0):
                tumors.append('No Tumor')
            
            else:
                tumor = f'{class_labels[pred_class]} tumor'# tumor type
                tumors.append(tumor)
            confidence = pred[pred_class] # how much confidence that is that type of tumor is 
            confidences.append(confidence)
                    

    with col2:
        st.subheader("Current Scan")
        curr_img_file = st.file_uploader("Upload Current MRI", key="curr")
       # model = load_model('/home/rgukt/Desktop/ML PROJECTS/Medical-Image-Analysis-Brain-Tumor-Detection-main/vgg_base_model.h5')
        if curr_img_file is not None:
            curr_img = Image.open(curr_img_file)
            st.image(curr_img,caption='uploaded image',use_container_width=True)
                
            
            test_img = preprocess_test_image(curr_img)
            pred = model.predict(test_img)[0]
            # st.write(pred)
            pred_class = np.argmax(pred)
            if(pred_class == 0):
                tumors.append('No Tumor')
            
            else:
                tumor = f'{class_labels[pred_class]} tumor'# tumor type
                tumors.append(tumor)
            confidence = pred[pred_class] # how much confidence that is that type of tumor is 
            confidences.append(confidence)
    if st.button('Compare'):
        if len(tumors) == 2 and len(confidences) == 2:
            df = pd.DataFrame(
                {
                    'Previous Scan': [tumors[0], confidences[0]],
                    'Current Scan': [tumors[1], confidences[1]]
                },
                index=['Tumor Type', 'Confidence']
            )
            st.dataframe(df)

            if tumors[0] == tumors[1]:
                trend_msg = confidence_trend(confidences[0], confidences[1])
                st.write(f"📌 AI Observation: {trend_msg}")
            else:
                st.warning(
                    "Tumor type prediction differs between scans. "
                    "Confidence comparison may not be reliable."
                )

            st.info(
                "Note: This comparison reflects only AI prediction confidence changes "
                "and does not indicate tumor improvement or worsening."
            )
        else:
            st.warning("Please upload both previous and current MRI scans before comparison.")


   
