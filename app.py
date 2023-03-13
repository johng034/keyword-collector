from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import pdfplumber
import pickle
import openai
import os

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
    return ' '.join(stopwords_removed)

def get_keywords(job_description):
    # generate candidate phrases using GPT-3
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Find the top 20 keywords separated by a comma in the following text:\n{job_description}\nKeywords:",
    max_tokens=150,
    # n=1,
    # stop=None,
    temperature=0.5,
    )

    # extract candidate phrases from response
    keywords = response.choices[0].text
    keywords = keywords.strip()
    keywords = keywords.split(', ')

    return keywords

def compare_texts(text1, text2):
    model_engine = "text-davinci-003"
    prompt = f"""
    Compare the similarity of two texts:\n\n
    Job Description: {text1}\n\n
    Resume: {text2}\n\n
    Give a score from 0 to 100 where 0 means the resume contains none of the keywords from the job description and 100 means the resume contains most of the keywords from the job description.
    Give a brief explanation why you gave that score without providing the keywords.
    Output your response as follows:

    Score: [score]\n
    Explanation: [Reason why you gave that score]
    """
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0.5,
        max_tokens=200,
        # n=1,
        # stop=None,
    )

    result = response.choices[0].text.strip()
    return result

# Get OpenAI's API key from environment variable
openai.api_key = os.environ['openai']

st.title('Resume Review')
st.markdown("""
            I created this project to help others navigate job descriptions and tailor their resumes.

            Please feel free to check out the [Github repository](https://github.com/johng034/keyword-collector).
            """)

## ----------------------------------------------------------------------------------------------
# JOB DESCRIPTION
job_description = st.text_input(label='Copy and paste the job description below:')
n_words = st.slider("Number of keywords", min_value=5, max_value=20, value=10)

if job_description != '':
    job_description_words = clean_text(job_description)
    job_description_cleaned = remove_stopwords(job_description_words)

## ----------------------------------------------------------------------------------------------
    ## NOTE:
    ## My first attempt used the CountVectorizer to calculate the word frequency and use the most frequent words as the keywords
    ## Issues arose when certain keywords were shown only once even though they were important to the role
    ## I will keep my original code here but comment it out

    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(job_description_cleaned)
    # words = vectorizer.get_feature_names_out()
    # frequency = X.toarray().sum(axis=0)

    # data = {
    #         'Keyword': words,
    #         'Frequency': frequency
    #         }

    # # Create a DataFrame 
    # df = pd.DataFrame(data=data)

    # # Sort the keywords by frequency and save the top 25%
    # words = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    # twenty_five_percent = int(len(words)*0.25)
    # top_keywords = words[:twenty_five_percent]
## ----------------------------------------------------------------------------------------------
    ## New Strategy
    ## I decided to utilize GPT-3 to generate the keywords
    keywords = get_keywords(job_description_cleaned)

    ## NOTE: 
    ## I used TF-IDF vectorizer to run a second analysis on the keywords, but it wasn't getting the results I expected

    # # create TF-IDF vectorizer and fit on input text
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    # vectorizer.fit([job_description_cleaned])

    # # transform candidate phrases into TF-IDF vectors
    # vectors = vectorizer.transform(keywords)

    # # compute TF-IDF scores for candidate phrases
    # scores = vectors.sum(axis=0)
    # scores = scores.tolist()[0]

    # # pair candidate phrases with their TF-IDF scores and sort by score
    # results = list(zip(vectorizer.get_feature_names(), scores))
    # results = sorted(results, key=lambda x: x[1], reverse=True)
    # st.dataframe({
    #     "Keyword": keywords
    #     })


    # Create button to see keywords
    if st.button(label='View Keywords'):
        df = pd.DataFrame({"Keyword": keywords[:n_words]})
        df.index += 1
        st.dataframe(df)


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
        for word in keywords[:n_words]:
            # Check if keyword is in the resume 
            if word in cleaned_text:
                n_keywords += 1
                included_words.append(word)
            # If the keyword is not in the resume
            else:
                missing_words.append(word)

        st.subheader('Results Based on Keywords')

        percentage = (n_keywords/n_words)*100
        st.subheader(f'{round(percentage, 1)}%')
        st.markdown(f'Your resume contains {n_keywords} of the top {n_words} keywords found in the job description.')
        st.markdown(f'The keywords **included** in your resume are: {", ".join([word.capitalize() for word in included_words])}')
        st.markdown(f'The following words are *missing* from your resume: {", ".join([word.capitalize() for word in missing_words])}')

        st.subheader('GPT-3 Rating')
        score = compare_texts(job_description_cleaned, cleaned_text)
        st.write(score)

    else:
        st.text('Please add a job description and upload your resume and try again.')