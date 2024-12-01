import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import os, json

# Configure Generative AI API
with open("key2.txt") as f:
    KEY = f.read().strip()
genai.configure(api_key=KEY)

config = genai.GenerationConfig(response_mime_type="application/json", temperature=0)

# Define generative models
model1 = genai.GenerativeModel(model_name='models/gemini-1.5-flash', 
    system_instruction="""
                              You are an AI assistant designed to help visually impaired individuals understand images. I will provide you with an image, and your task is to describe it in detail. Your description should be:

                                Comprehensive - Explain everything visible in the image, including objects, people, scenery, and activities.
                                Contextual - Infer possible purposes or scenarios related to the image.
                                Sensory - Use language that conveys a sense of atmosphere, emotions, and the environment.
                                Actionable - Include additional details or explanations that would help a blind person better understand and imagine the image.
                              
                              Example Output:
                                For an image of a park:
                                Summary: The image shows a bustling park on a sunny afternoon, filled with people enjoying outdoor activities.
                                
                                Detailed_Description:The park is vast and green, with well-maintained grass and tall trees providing shade. Children are playing on a colorful playground set that includes swings, slides, and monkey bars. Nearby, families are seated on blankets, having picnics with baskets of food and drinks. Joggers and cyclists are using a paved path that winds through the park. A small pond is visible in the background, with ducks swimming and a couple sitting on a bench, feeding them. The sky is clear blue, with a few fluffy clouds, and the sunlight creates a warm and cheerful atmosphere.
                              
                              """)
    
model2 = genai.GenerativeModel(model_name='models/gemini-1.5-flash', 
    system_instruction="""
                              You are an AI assistant designed to help visually impaired individuals by extracting text from images. I will provide you with an image, and your task is to extract only the text visible in the image.
                              
                              """)

# Functions for tasks
def describe_image(img):
    img_array = np.array(img)
    img = Image.fromarray(img_array)
    response = model1.generate_content([img, "Describe the image"], generation_config=config)
    return response.text

def img_to_txt(img):
    img_array = np.array(img)
    img = Image.fromarray(img_array)
    response = model2.generate_content([img, "What is written in the image?"], generation_config=config)
    return response

def img_to_speech(img):
    extracted_text = img_to_txt(img)
    dict_response = json.loads(extracted_text.text)
    extracted_text = dict_response["text"]
    extracted_text = extracted_text.replace('\n', ' ')

    if extracted_text.strip():
        tts = gTTS(text=extracted_text, lang='en')
        audio_path = "output_audio.mp3"
        tts.save(audio_path)
        return audio_path
    else:
        return None

st.title("Assistive Image Processing Tool")
st.write("Upload an image and select an action to help visually impaired individuals.")

uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width =False, width=300)
    
    action = st.selectbox("Choose an action", ["None", "Describe the image", "Convert to Speech"])
 
    if action == "Describe the image":
        st.write("Describing the image...")
        description = describe_image(img)
        dict_response = json.loads(description)
        st.subheader("Image Description:")
        st.subheader("Summary")
        st.write(dict_response["Summary"])

        st.subheader("Detailed Description")
        st.write(dict_response["Detailed_Description"])
    
    elif action == "Convert to Speech":
        st.write("Converting text to speech...")
        audio_file = img_to_speech(img)
        if audio_file:
            st.audio(audio_file, format="audio/mp3")
        else:
            st.error("No text was extracted to convert to speech.")
  



