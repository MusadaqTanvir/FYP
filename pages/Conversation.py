import streamlit as st
import torch 
from nltk_utils import * 
from mymodel import *
import json 
import random 
import time 

st.markdown("""
            <style>
            .css-102x5pl.e1fqkh3o6
            {
            background-color: #27ae60;
            text-align: center;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.3);
            padding-left:50px;
            border: solid 1px white;
            border-radius: 20px;
            font-size: 20px;
            font-style: Arial;
            }
            .css-6qob1r.e1fqkh3o3
            {
            background-color: #454e56;
            }
            .main.css-k1vhr4.egzxvld5
            {
                background-color:;
            }
            .css-10pw50.egzxvld1
            {
                visibility: hidden;
            }
            .css-1uy0bt2.e1fqkh3o6{
                background-color:black;
                border: 1px solid white;
                border-radius: 20px;
                padding-left: 90px;
                text-decoration: bold;
                font-size:20px;
            }
            .css-6qob1r.e1fqkh3o3
            {
                background-color:blueviolet;
            }
            </style>
            """,unsafe_allow_html=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json','r') as f:
    intents = json.load(f)
FILE = 'data.pth'
data = torch.load(FILE)
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
total_words = data['total_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()
count = 0
bot_name = "SAM"
st.write("Let's Chat!, type 'quit' to exit")
while True:
    sentence = st.text_input("You: ", key=count)
    if not sentence:
        
        time.sleep(1000)
    elif sentence == 'quit':
        break
    else: 
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, total_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    st.text(f"{bot_name}: {random.choice(intent['responses'])}")
                    count = count + 1
        else:
            st.text(f"{bot_name} I do not understand...")
            count = count + 1
