from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pdfplumber

st.set_page_config(page_title='Keyword Finder')

def clean_text(text):
    text = re.sub('\n', ' ', text)  # Remove '\n'
    text = text.strip()  # Remove whitespaces
    text = re.sub("'", '', text)  # Remove ' with no spaces (e.g., they're -> theyre)
    text = re.sub(",", '', text)  # Remove , with no spaces (e.g., 10,000 -> 10000)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    text = text.lower()  # Lowercase all the text
    text = text.strip()  # Remove any final whitespaces
    return text

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    text = text.split()
    # text = ['The']
    stopwords_removed = [word for word in text if word not in stopwords]
    return stopwords_removed
st.title('Keyword Finder')
st.markdown("""
            I created this project to help others navigate job descriptions and tailor their resumes.

            Please feel free to check out the [Github repository](https://github.com/johng034/keyword-collector).
            """)

job_description = st.text_input(label='Copy and paste the job description below:')

if job_description != '':
    job_description_cleaned = clean_text(job_description)
    word_list = remove_stopwords(job_description_cleaned)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(word_list)

    # Create a DataFrame 
    df = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names_out())

    # Sort the values
    words = df.sum(axis=0).sort_values(ascending=False)

    # Create button to see keywords
    if st.button(label='View Keywords'):
        st.dataframe(words[:25])




# Upload a file (PDF)
st.subheader('Compare your resume to the job description.')
uploaded_file = st.file_uploader(label='Upload your resume', type='pdf')

if st.button(label='Compare'):
    if uploaded_file != None:
        with pdfplumber.open(uploaded_file) as pdf:

            # Go through each page and extract text
            complete_resume = []
            for page in pdf.pages:
                pdf_content = page.extract_text()
                complete_resume.append(pdf_content)

        # Save complete text
        resume_text = '\n'.join(complete_resume)
        cleaned_text = clean_text(resume_text)
        st.markdown(cleaned_text.strip())

        ## TODO Count the number of instances a keyword is in the resume
        ## TODO Weigh the most frequent keywords more

    else:
        st.text('Please upload a file and try again.')