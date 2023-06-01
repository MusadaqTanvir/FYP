import streamlit as st
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
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
            </style>
            """,unsafe_allow_html=True)
# Lets Train the model first of all...............................................

Model_link =f"cardiffnlp/twitter-roberta-base-sentiment"
Tokenizer = AutoTokenizer.from_pretrained(Model_link)
Model_Obj = AutoModelForSequenceClassification.from_pretrained(Model_link)

score_result = {}
text = ''
choice = st.radio('***Select the Method***',('File','Text'))
st.markdown("---")

if choice == 'Text':
    text = st.text_input("Enter Your Query Please")
    encoded_text = Tokenizer(text,return_tensors='pt')
    output_score = Model_Obj(**encoded_text)
    score = output_score[0][0].detach().numpy()
    score = softmax(score)
    score_result = {"Negative":score[0],'Neutral':score[1],'Positve':score[2]}

elif choice == 'File':
    file = st.file_uploader("***Please Upload Your File!*** ")
    try:
        data = pd.read_csv(file).head(500)
        if len(data) > 500:
            st.write("We are reducing the size to 500 records")
            data = data[:500]
            st.dataframe(data.head())
        else:
            st.dataframe(data.head())
        def sentiment_Analysis_Roberta(text):
            encoded_text = Tokenizer(text,return_tensors='pt')
            output_score = Model_Obj(**encoded_text)
            score = output_score[0][0].detach().numpy()
            score = softmax(score)
            return {'Negative':score[0],'Neutral':score[1],'Positive':score[2]}
        result = {}
        test_data = data[['Id','Text']]
        Total_result = {"Negative":[],"Neutral":[],"Positive":[]}
        for MyId, text in zip(test_data['Id'],test_data['Text']):
            try:
                result[MyId] = sentiment_Analysis_Roberta(text)
                Total_result[max(result[MyId],key=result[MyId].get)].append(max(result[MyId].values()))
            except RuntimeError:
                print("Token Error")
        bar_data = {'Negative':len(Total_result['Negative']),'Neutral':len(Total_result['Neutral']),'Positive':len(Total_result['Positive'])}
        bar_data = pd.DataFrame(bar_data,index=[1])
        
        fig, ax = plt.subplots(figsize=(4,3), layout='constrained')
        cols = bar_data.columns
        vals = bar_data.iloc[0,:]
        ax.bar(cols,vals,color=['red','blue','green'])
        st.pyplot(fig,use_container_width=False)
    except ValueError:
        st.warning("File Seems to Missing")
        st.markdown("---")
st.write("Your Answer has Following Semantics")
if text == '':
    st.warning("Data Not Inserted Yet")
else:
    st.write(score_result)
