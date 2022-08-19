from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re

st.set_page_config(page_title='Keyword Collector')

st.title('hi steph ur cute')
st.markdown("""
            I created this project to help others navigate job descriptions and tailor their resumes.

            Please feel free to check out the [Github repository](https://github.com/johng034/keyword-collector).
            """)

def clean_text(text):
    text = text.strip()  # Remove whitespaces
    text = re.sub('\n', '', text)  # Remove '\n'
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    text = text.lower()  # Lowercase all the text
    text = text.strip()  # Remove any final whitespaces
    return text

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    text = text.split()
    stopwords_removed = [word for word in text if word not in stopwords]
    return stopwords_removed

job_description = st.text_input(label='Please copy and paste the job description below:')

if job_description is not '':
    job_description_cleaned = clean_text(job_description)
    word_list = remove_stopwords(job_description_cleaned)

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(word_list)

    # Create a DataFrame 
    df = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())

    # Sort the values
    words = df.sum(axis=0).sort_values(ascending=False)

    st.dataframe(words[:25])