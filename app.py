from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pdfplumber
import pickle

st.set_page_config(page_title='Resume Review')

def clean_text(text):
    text = re.sub('\n', ' ', text)  # Remove '\n'
    text = text.strip()  # Remove whitespaces
    text = text.lower()  # Lowercase all the text
    text = re.sub("'", '', text)  # Remove ' with no spaces (e.g., they're -> theyre)
    text = re.sub(",", '', text)  # Remove , with no spaces (e.g., 10,000 -> 10000)
    text = re.sub("etc", '', text)  # Remove , with no spaces (e.g., 10,000 -> 10000)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    text = text.strip()  # Remove any final whitespaces
    return text

def remove_stopwords(text):
    with open ('stopwords', 'rb') as file:
        stopwords = pickle.load(file)
    text = text.split()
    stopwords_removed = [word for word in text if word not in stopwords]
    return stopwords_removed
st.title('Resume Review')
st.markdown("""
            I created this project to help others navigate job descriptions and tailor their resumes.

            Please feel free to check out the [Github repository](https://github.com/johng034/keyword-collector).
            """)

## ----------------------------------------------------------------------------------------------
# JOB DESCRIPTION
job_description = st.text_input(label='Copy and paste the job description below:')

if job_description != '':
    job_description_cleaned = clean_text(job_description)
    word_list = remove_stopwords(job_description_cleaned)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(word_list)

    words = vectorizer.get_feature_names_out()
    frequency = X.toarray().sum(axis=0)

    data = {
            'Keyword': words,
            'Frequency': frequency
            }

    # Create a DataFrame 
    df = pd.DataFrame(data=data)

    # Sort the keywords by frequency and save the top 25%
    words = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    twenty_five_percent = int(len(words)*0.25)
    top_keywords = words[:twenty_five_percent]

    # Create button to see keywords
    if st.button(label='View Keywords'):
        st.dataframe(top_keywords)


## ----------------------------------------------------------------------------------------------
# RESUME
# Upload a file (PDF)
st.subheader('Compare your resume to the job description.')
uploaded_file = st.file_uploader(label='Upload your resume', type='pdf')

## ----------------------------------------------------------------------------------------------
# RESULTS
if st.button(label='Compare'):
    if uploaded_file != None and job_description != '':
        with pdfplumber.open(uploaded_file) as pdf:

            # Go through each page and extract text
            complete_resume = []
            for page in pdf.pages:
                pdf_content = page.extract_text()
                complete_resume.append(pdf_content)

        # Collect and clean the text from the resume
        resume_text = '\n'.join(complete_resume)
        cleaned_text = clean_text(resume_text)

        # Calculate the similarity between the resume and job description
        n_keywords = 0
        included_words = []
        missing_words = []

        # Iterate through top job description keywords
        for word in top_keywords['Keyword']:
            # Check if keyword is in the resume 
            if word in cleaned_text.split():
                n_keywords += 1
                included_words.append(word)
            # If the keyword is not in the resume
            else:
                missing_words.append(word)

        st.subheader('Results')

        percentage = (n_keywords/twenty_five_percent)*100
        st.subheader(f'{round(percentage, 1)}%')
        st.markdown(f'Your resume contains {n_keywords} of the top {twenty_five_percent} keywords found in the job description.')
        st.markdown(f'The keywords **included** in your resume are: {", ".join([word.capitalize() for word in included_words])}')
        st.markdown(f'The following words are *missing* from your resume: {", ".join([word.capitalize() for word in missing_words])}')

        ## TODO Place a higher weight on the most frequent keywords 

    else:
        st.text('Please add a job description and upload your resume and try again.')