import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
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


files = st.file_uploader("Upload Files Minimum 2 Files", accept_multiple_files=True)
if len(files) > 1:
    try:
        names = [f.name for f in files]
        #st.write(names)
        contents = [file.read().decode('utf-8') for file in files]
        def Vectorizer(text):
            return TfidfVectorizer().fit_transform(text).toarray()

        def Similarity(doc1, doc2):
            return cosine_similarity([doc1, doc2])
        
        vectors = Vectorizer(contents)
        #st.write(vectors)
        named_vectors = list(zip(names, vectors))
        #st.write(named_vectors)
        #st.write(named_vectors)
        Plagerism_result = set()
        def Plagerism():
            global named_vectors
            #st.write(named_vectors)
            new_vectors = named_vectors.copy()
            #st.write(new_vectors)
            for name_a, vector_a in named_vectors:
                current_index = new_vectors.index((name_a, vector_a))
                #st.write(current_index)
                del new_vectors[current_index]
                for name_b, vector_b in new_vectors:
                    sim_score = Similarity(vector_a, vector_b)[0][1]
                    vector_pair = sorted((name_a, name_b))
                    score = (vector_pair[0], vector_pair[1], sim_score)
                    #st.write(score)
                    Plagerism_result.add(score)
            return st.dataframe(Plagerism_result)
        if st.button("Results"):
            Plagerism()
    except ValueError:
        st.write("Files Not Uploaded Yet")
else:
    st.write("Minimum Files should be 2")