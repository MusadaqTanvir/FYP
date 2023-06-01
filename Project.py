# Importing the Libraries.....
import streamlit as st
#--------------------------
# settig the App main page layout to wide...
st.set_page_config(layout='wide')
# ---------------------------------------
# reading the CSS(cascading style sheet file) ....
with open('style.css') as file:
    st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
#---------------------------------------------------------------------
# App main heading starting...
st.header("Smart Chat and Text Analyzer")
st.image('Robot2.jpg', width=700)


st.sidebar.markdown("***")
