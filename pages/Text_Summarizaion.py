import streamlit as st
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
st.markdown("""
            <style>
            .css-102x5pl.e1fqkh3o6
            {
            background-color: #27ae60;
            text-align: center;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.3);
            padding-left:25px;
            border: solid 1px white;
            border-radius: 20px;
            font-size: 20px;
            font-style: Arial;
            }
            .css-6qob1r.e1fqkh3o3
            {
                background-color: #454e56;
            }
            .css-10pw50.egzxvld1
            {
                visibility: hidden;
            }
                        .css-6qob1r.e1fqkh3o3
            {
                background-color:blueviolet;
            }
            .css-1uy0bt2.e1fqkh3o6{
                background-color:black;
                border: 1px solid white;
                border-radius: 20px;
                padding-left: 70px;
                text-decoration: bold;
                font-size:20px;
            }
           block-container.css-z5fcl4.egzxvld4{
               border:1px solid white;
           }
            </style>
            """,unsafe_allow_html=True)

def text_summarizer(text):
    st.write(text)
    st.markdown("---")
    stopwords =''
    punctuation = ''
    stopwords = list(STOP_WORDS)
    punctuation = list(punctuation)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    token = [token.text for token in doc]
    #st.write(token)
    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text.lower() not in word_freq.keys():
                word_freq[word.text.lower()] = 1
            else:
                word_freq[word.text.lower()] += 1
    if len(word_freq) == 0:
        st.warning("Empty File")
    else:
        max_freq = max(word_freq.values())
        for word in word_freq.keys():
            word_freq[word] = word_freq[word]/max_freq
        sent_tokens = [sent for sent in doc.sents]
        sent_scores = {}
        for sent in sent_tokens:
            for word in sent:
                if word.text.lower() in word_freq.keys():
                   if sent not in sent_scores.keys():
                      sent_scores[sent] = word_freq[word.text.lower()]
                   else:
                        sent_scores[sent] += 1
        select_length = int(len(sent_tokens)* 0.3)
        summary = nlargest(select_length,sent_scores,key=sent_scores.get)
        final_summary = [word.text for word in summary]
        summary = ' '.join(final_summary)
        Total_length = len(text.split(' '))
        Summarized_length = len(summary.split(' '))
        st.write(summary)
        st.write(Total_length)
        st.write(Summarized_length)
        st.markdown("---")








answer = st.radio("Select Your Input Method Pleae",('File','Simple Text'))
if answer == 'File':
   st.write("Hello World")
   contents = ''
   file1  = st.file_uploader("Upload Your Text File Here(.txt format ) ")
   try:   
       text = file1.read()
       text = text.decode('utf-8')
       text_summarizer(text)
       
   except AttributeError:
       st.warning("File Not Found")

       
if answer == 'Simple Text':
    text = st.text_area("Paste Here Your text Please: ")
    if text != None:
        text_summarizer(text)
