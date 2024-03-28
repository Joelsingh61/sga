from dotenv import load_dotenv

load_dotenv() ## load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load generative AI model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

# Display file uploader widget
csv_file = st.file_uploader("Upload CSV file", type=['csv'])

if csv_file is not None:
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Display DataFrame
    st.write(df)

    # Analyze data with generative AI model
    for column in df.columns:
        st.write(f"Analyzing column: {column}")
        for index, row in df.iterrows():
            input_text = row[column]
            encoded_input = tokenizer.encode(input_text, return_tensors='pt')
            output = model.generate(encoded_input, max_length=100)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.write(generated_text)

def get_gemini_repsonse(input,image,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,image[0],prompt])
    return response.text



def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
##initialize our streamlit app

st.set_page_config(page_title="Gemini Health App")

st.header("Gemini Health App")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Analyse the Csv File")

input_prompt="""
You are an expert in student grade analysing u need to analyse all the grades of stuents and say 
u need to analys the uploaded csv file and u need to give a proper report dont give wronng answers

               
               ----
               ----


"""

## If submit button is clicked

if submit:
    image_data=input_image_setup(uploaded_file)
    response=get_gemini_repsonse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)
