import streamlit as st
import pandas as pd
from PIL import Image
# Title for the file uploader widget
st.title('Upload a CSV file')
# Create a file uploader widget
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the DataFrame
    st.write(df)


# Load CSV file
csv_file = st.file_uploader("Upload CSV file", type=['csv'])

if csv_file is not None:
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Display DataFrame
    st.write(df)

    # Convert DataFrame to an image (optional)
    img = Image.frombytes('RGB', (800,600), df.to_string().encode('utf-8'))

    # Display the image
    st.image(img, caption='CSV data as image')

