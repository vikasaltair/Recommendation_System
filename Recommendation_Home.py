#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import os
import shutil

dir = "Intermediate_Folder"
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

st.set_page_config(
    page_title="Recommendation Home"
)

st.write("""
         ## Product Recommendation
         """)
#st.markdown("<p style='font-family:cursive;font-size: 15px;'>Recommendation App developed by [**Blue Altair's DS Team**](https://www.bluealtair.com/)</p>", unsafe_allow_html=True)
st.caption("Recommendation App developed by [**Blue Altair's DS Team**](https://www.bluealtair.com/)")
st.sidebar.success("Choose Recommendation Type")
st.markdown(
    """
    Popular platforms like Netflix, Spotify, YouTube, Amazon etc. recommends items as per your interest and preference by analyzing your past interaction or behavior with the system.
    
    **Product recommendations** can help in:
    - Personalized Recommendations
    - Engaging the customers
    - Boosting sales and revenue
    - Delivering the most relevant content
    - Maintaining the brand experience
    
    Broadly speaking there are two kinds of recommendation approaches:
    1. **Content-based recommendations** - recommending based on the content(metadata) about the products/items, for e.g. product title, description, images, category/subcategory, specification, etc. 
    2. **Collaborative Filtering** - Focuses more on users past behaviour/preference. It filters out the products that a user might like on the basis of reactions by similar users.
        - Collaborative Filteting can be further broadly classified into two categories.
          - **Similarity based methods** – User-user, Item-item similarity based on approaches like Nearest neighbors, kNN etc.
          - **Matrix Factorization** - SVD(Singular Value Decomposition), NMF(Non- negative Matrix Factorization), ALS(Alternating Least Square)
    """)

