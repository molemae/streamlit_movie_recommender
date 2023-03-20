import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from  recommender import recommend_random, recommend_with_NMF, recommend_neighborhood, ratings



st.markdown(f"""
    <style>
    .stApp{{
        background-image: url('https://smit.vub.ac.be/wp-content/uploads/2021/11/felix-mooneeram-evlkOfkQ5rE-unsplash-scaled.jpg');
        background-size: cover;
    }}
    </style>
    """,unsafe_allow_html=True)


nav = st.sidebar.radio(
    " ",
    ["Home", "Wombats"]
    ) 

if nav == "Home":
    st.title(":red[Welcome to Simoritztinis Movie Recommender - Streamlit Edition]")
    st.image("static/cat2.gif")
    st.markdown(' ')
    st.markdown(' ')
    model_choice = st.selectbox('Select Recommedation Model', ['Random', 'NMF','Cosine Similarity'])
    #pred = st.button('Get Recommendations', on_click(get_rec = True))
   
    if model_choice !='Random':
        col1, col2, col3, col4 = st.columns(4)
        movie0 = col2.selectbox(":red[First movie]", ratings.columns)
        rating0 = col3.number_input(":red[Rating for First movie]",
            min_value = 1,
            max_value = 5)
        col5, col6, col7, col8 = st.columns(4)
        movie1 = col6.selectbox(":red[Second movie]", ratings.columns, key = "second movie")
        rating1 = col7.number_input(":red[Rating for Second movie]",
            min_value = 1,
            max_value = 5,
        key = "rating for second movie")
        col9, col10, col11, col12 = st.columns(4)
        movie2 = col10.selectbox(":red[Third movie]", ratings.columns, key = "third movie")
        rating2 = col11.number_input(":red[Rating for Third movie]",
            min_value = 1,
            max_value = 5)
    
        user_input = {
            "movie0": movie0,
            "rating0": int(rating0),
            "movie1": (movie1),
            "rating1": int(rating1),
            "movie2": movie2,
            "rating2": int(rating2),
            }



    if st.button('Get recommendation'):
        if model_choice == 'Random':
            st.write(':red[This is Random]')
            st.write(pd.DataFrame(recommend_random()))
        elif model_choice == 'NMF':
            st.write(':red[This is NMF]')
            st.write(pd.DataFrame(recommend_with_NMF(query=user_input)))
        else:
            st.write(':red[This is Cosine]')
            st.write(pd.DataFrame(recommend_neighborhood(query=user_input)))




if nav == "Wombats":
    st.title(':blue[Ciao Cacao]')
    st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc.tenor.com%2Fv_CRiqOnPpIAAAAC%2Fwombat-eating.gif&f=1&nofb=1&ipt=7f5599fb69eee74da6e82f3972ba281588135a4e457d99b88cb78e8ae2f11636&ipo=images')
    st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia1.tenor.com%2Fimages%2F19994749f3c8f7574ab5ddf8b925a1ec%2Ftenor.gif%3Fitemid%3D8758672&f=1&nofb=1&ipt=c0bb654cab47fddbfe7c4a87bc37ffaaceb2b47a56d6520de8a4f4658c51d83a&ipo=images')
    st.image('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmedia.giphy.com%2Fmedia%2F14miSV6VMiO7te%2Fgiphy.gif&f=1&nofb=1&ipt=2dce481028513f4fe070a6f0d491621c06a793ec8f85e93f899730432dfe9ece&ipo=images')