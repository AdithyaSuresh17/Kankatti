from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import os

# Load environment variables
load_dotenv()

# Import Generative AI library
# Update the import statement based on the correct library you're using
import google.generativeai as genai

# Configure Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create GenerativeModel object
model = genai.GenerativeModel("gemini-pro-vision")

# Function to get response from Gemini model
def get_gemini_response(input_text, image):
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# Set Streamlit page config
st.set_page_config(page_title="Image Describer")

# Streamlit UI
st.header("Gemini Image Describer")

input_text = st.text_input("Input:")
uploaded_file = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

submit = st.button("Describe the image")

if submit:
    response = get_gemini_response(input_text, image)
    st.subheader("The response is:")
    st.write(response)
