import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

import pickle

from  recommender import recommend_random, recommend_with_NMF, recommend_neighborhood



"data/nmf_model2.pkl"
    
st.markdown(f"""
    <style>
    .stApp{{
        background-image: url(data:'static/felix-mooneeram-evlkOfkQ5rE-unsplash.jpg');
        background-size: cover;
    }}
    </style>
    """,unsafe_allow_html=True)


nav = st.sidebar.radio(
    " ",
    ["Home", "Recommended"]
    ) 

if nav == "Home":
   st.title("Welcome to Moe's Movie Recommender - Streamlit Edition")
   st.image("static/cat2.gif", width=400)
   
   model_choise = st.selectbox('Select Recommedation Model', ['Random', 'NMF','Cosine Similarity'])

   if model_choise == 'Random':
        st.write('random')
        st.write(recommend_random())
   if model_choise == 'NMF':
        st.write('NMF')
   else:
    st.write('Cosine')


   st.markdown(
       'Button and boxes'
    )

if nav == "Recommended":
    st.write("Here are your recommendations")
#     st.write("Welcome to the section on Exploratory Data Analysis.")
#     st.write("Data is taken from [titanic](https://www.kaggle.com/competitions/titanic/data?select=train.csv)")
#     if st.checkbox("Click here to see the original dataframe"):
#         st.title("Titanic training dataset")
#         st.dataframe(df)
#     if st.checkbox("Click here to see missing values"):
#         fig,ax = plt.subplots()
#         st.title("Heatmap features titanic dataset")
#         sns.heatmap(df.corr(), ax = ax)
#         st.write(fig)
#     if st.checkbox("Click here to see distribution of variables"):
#         st.title("Distribution features titanic dataset")
#         fig = sns.pairplot(df, hue = "Survived")
#         st.pyplot(fig)
#     if st.checkbox("Click here to see the variables taken into account for model fit"):
#         st.title("Features for model fit")
#         st.write(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].columns)

# if nav == "Prediction":
#     st.markdown(
#     """ #### Welcome to the predictions page.
#     """
#     )
   
#     st.markdown(
#     """ ##### For predictions regarding if a passenger survived or not, please answer the following questions. 
#     """
#     )

#     @st.cache
#     def load_model():
#         with open(MODEL_FILE, "rb") as file_in:
#             clf_LR = pickle.load(file_in)
#         return clf_LR

#     clf_LR = load_model()

#     col1, col2, col3 = st.columns(3)

#     gender = col1.number_input(
#         "Gender (Type '1' for 'female' and '0' otherwise)", 
#         min_value = 0,
#         max_value = 1,
#         )
#     age = col2.number_input(
#         "Age", 
#         min_value = 1,
#         max_value = 100,
#         )
#     siblings = col3.slider(
#         "How many siblings does the passenger have? (Choose between '0' and '8')", 
#         min_value = 0,
#         max_value = 8,
#         step = 1
#         )

#     col4, col5, col6 = st.columns(3)

#     parch = col4.number_input(
#         "How many family member (parents/children) were aboard?", 
#         min_value = 1,
#         max_value = 10,
#         )
#     female_pclass = col5.slider(
#         "If female: which class did the passenger travel in?", 
#         min_value = 0,
#         max_value = 3,
#         step = 1
#         )
#     male_pclass = col6.slider(
#         "If male: which class did the passenger travel in?", 
#         min_value = 0,
#         max_value = 3,
#         step = 1,
#         key=101
#         )

#     user_input = {
#             "gender": int(gender),
#             "age": int(age),
#             "siblings": int(siblings),
#             "parch": int(parch),
#             "female_pclass": int(female_pclass),
#             "male_pclass": int(male_pclass),
#             }

#     user_input = pd.DataFrame(
#         user_input, 
#         columns = user_input.keys(), 
#         index = [0])

#     user_input = np.array(user_input).reshape(1, -1)

#     pred = clf_LR.predict(user_input)[0]
#     proba = clf_LR.predict_proba(user_input)[0]

#     if st.button("Predict"):
#         st.success(f'This person would have survived with a probability of: {round(proba[1], 2)}')




