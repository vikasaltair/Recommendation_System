#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from io import StringIO
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import os
from os import path

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
distil_bert = 'distilbert-base-uncased'

from sklearn.preprocessing import OneHotEncoder 
# In[ ]:


def preprocessing(df, col):
    df = df.dropna() # excluding the records with missing values
    df.sort_values(col, inplace=True, ascending=False)
    duplicated_series = df.duplicated(col, keep = False)
    df = df[~duplicated_series] # removing duplicate records
    df = df.reset_index(drop=True)
    return df


# In[ ]:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


stop_words = set(stopwords.words('english'))


# In[ ]:


def stopwords_removal(df, col):
    for i in range(len(df[col])):
        string = ""
        for word in df[col][i].split():
            word = ("".join(e for e in word if e.isalnum()))
            word = word.lower()
            if not word in stop_words:
                string += word + " " 
        df.at[i,col] = string.strip()
    return df


# In[ ]:


lemmatizer = WordNetLemmatizer()


# In[ ]:


def lemmatization(df, col):
    for i in range(len(df[col])):
        string = ""
        for w in word_tokenize(df[col][i]):
            string += lemmatizer.lemmatize(w,pos = "v") + " "
        df.at[i, col] = string.strip()
    return df


# In[ ]:


def text_cleaning(df, col):
    df = stopwords_removal(df, col)
    df = lemmatization(df, col)
    return df


# In[ ]:


def fe_using_BoW(df, col):
    BoW_vectorizer = CountVectorizer()
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}")==False):
        os.makedirs(f"{root_dir}/{dir}/{fe_tech}")
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}/BoW_features.pickle")==False):
        features = BoW_vectorizer.fit_transform(df[col])
        pickle.dump(BoW_vectorizer, open(f"{root_dir}/{dir}/{fe_tech}/BoW_features.pickle", "wb"))
    else:
        BoW_vectorizer = pickle.load(open(f"{root_dir}/{dir}/{fe_tech}/BoW_features.pickle","rb"))
        features = BoW_vectorizer.transform(df[col])
    return features



# In[ ]:


def fe_using_TF_IDF(df, col):
    tfidf_vectorizer = TfidfVectorizer(min_df = 0)
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}")==False):
        os.makedirs(f"{root_dir}/{dir}/{fe_tech}")
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}/tfidf_features.npy")==False):
        features = tfidf_vectorizer.fit_transform(df[col])
        pickle.dump(tfidf_vectorizer, open(f"{root_dir}/{dir}/{fe_tech}/tfidf_vectorizer.pickle", "wb"))
    else:
        tfidf_vectorizer = pickle.load(open(f"{root_dir}/{dir}/{fe_tech}/tfidf_vectorizer.pickle","rb"))
        features = tfidf_vectorizer.transform(df[col])
    return features


# In[ ]:


def fe_using_Word2Vec(df, col):
    with open('word2vec_model', 'rb') as file:
        loaded_model = pickle.load(file)
    vocabulary = loaded_model.keys()
    w2v = []
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}")==False):
        os.makedirs(f"{root_dir}/{dir}/{fe_tech}")
    if(os.path.exists(f"{root_dir}/{dir}/{fe_tech}/word2vec_features.npy")==False):
        for i in df[col]:
            w2Vec_word = np.zeros(300, dtype="float32")
            for word in i.split():
                if word in vocabulary:
                    w2Vec_word = np.add(w2Vec_word, loaded_model[word])
            w2Vec_word = np.divide(w2Vec_word, len(i.split()))
            w2v.append(w2Vec_word)
        w2v = np.array(w2v)
        np.save(f"{root_dir}/{dir}/{fe_tech}/word2vec_features.npy", w2v)
    else:
        w2v = np.load(f"{root_dir}/{dir}/{fe_tech}/word2vec_features.npy", allow_pickle=True)
    return w2v


max_len = 50
def fe_using_Distil_BERT(df, col):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    if(os.path.exists(f"{root_dir}/{dir}/BERT")==False):
        os.makedirs(f"{root_dir}/{dir}/BERT")
    if(os.path.exists(f"{root_dir}/{dir}/BERT/distil_bert_features.npy")==False):
        tokenized = df[col].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids =tf.convert_to_tensor(padded)  
        attention_mask = tf.convert_to_tensor(attention_mask)
        last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:,0,:].numpy()
        np.save(f"{root_dir}/{dir}/BERT/distil_bert_features.npy", features)
    else:
        features = np.load(f"{root_dir}/{dir}/BERT/distil_bert_features.npy", allow_pickle=True)
    return features

# In[ ]:
def feature_featurization(df, feature, product_index):
    if(is_object_dtype(df[feature])):
        feature_onehot_encoded = OneHotEncoder().fit_transform(np.array(df[feature]).reshape(-1,1))
        feature_dist = pairwise_distances(feature_onehot_encoded, feature_onehot_encoded[product_index]) + 1  
    else:
        feature_dist = pairwise_distances(df[feature].values.reshape(-1,1), pd.Series(df[feature][product_index]).values.reshape(1,-1))
    return feature_dist
def model(product_id, col, w1=1, w2 = 0, w3 = 0, num_similar_items=10):
    df_dummy = pd.DataFrame()
    if(is_numeric_dtype(st.session_state.df_inp[prod_id])):
        product_id = int(product_id)
    if(fe_tech=='Bag of Words'):
        features = fe_using_BoW(st.session_state.df_temp, col)
    elif(fe_tech=='TF-IDF'):
        features = fe_using_TF_IDF(st.session_state.df_temp, col)
    elif(fe_tech=='Word2Vec'):
        features = fe_using_Word2Vec(st.session_state.df_temp, col)
    elif(fe_tech=='Distil BERT'):
        features = fe_using_Distil_BERT(st.session_state.df_temp, col)    
    product_index = st.session_state.df_temp[st.session_state.df_temp[prod_id]==product_id].index.values[0]
    if(fe_tech not in ['Word2Vec', 'Distil BERT']):
        couple_dist = pairwise_distances(features, features[product_index])
    else:
        couple_dist = pairwise_distances(features, features[product_index].reshape(1,-1))
    if(feature2 in st.session_state.df_inp.columns.tolist()):
        feature2_dist = feature_featurization(st.session_state.df_temp, feature2, product_index) 
    if(feature3 in st.session_state.df_inp.columns.tolist()):
        feature3_dist = feature_featurization(st.session_state.df_temp, feature3, product_index) 
    if(feature2 in st.session_state.df_inp.columns.tolist() and feature3 in st.session_state.df_inp.columns.tolist()):
        weighted_couple_dist = (float(w1) * couple_dist +  float(w2) * feature2_dist + float(w3) * feature3_dist)/(float(w1) + float(w2) + float(w3))
        indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        if(w2>0 and w3>0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title, feature2, feature3]]
        elif(w2==0 and w3==0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title]]
        elif(w2==0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title, feature3]]
        elif(w3==0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title, feature2]]
    elif(feature2 in st.session_state.df_inp.columns.tolist()):
        weighted_couple_dist = (float(w1) * couple_dist +  float(w2) * feature2_dist)/(float(w1) + float(w2))
        indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        if(w2==0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title]]
        else:
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title, feature2]]
    elif(feature3 in st.session_state.df_inp.columns.tolist()):
        weighted_couple_dist = (float(w1) * couple_dist +  float(w3) * feature3_dist)/(float(w1) + float(w3))
        indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        if(w3==0):
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title]]
        else:
            df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title, feature3]]
    else:
        weighted_couple_dist = couple_dist
        indices = np.argsort(weighted_couple_dist.ravel())[0:num_similar_items]
        df_dummy = st.session_state.df_inp.loc[indices, [prod_id, title]]
    df_dummy["Similarity Score"] = 1 / (1+weighted_couple_dist[indices].ravel())
    with st.container(): 
        st.write("**Queried product details**")
        st.write(f'{prod_id} : ',df_dummy[prod_id].iloc[0])
        st.write(f'{title} : ',df_dummy[title].iloc[0])
        if(feature2 in st.session_state.df_inp.columns.tolist()):
            if(w2>0):
                st.write(f'{feature2} : ',df_dummy[feature2].iloc[0])
        if(feature3 in st.session_state.df_inp.columns.tolist()):
            if(w3>0):
                st.write(f'{feature3} : ',df_dummy[feature3].iloc[0])
        st.markdown("**Recommended product**")
        styler = df_dummy.iloc[1:,].style.hide_index()
        st.write(styler.to_html(index = False), unsafe_allow_html=True)
    #recommend_all(col, w1=10, w2 = 1, w3 = 1, num_similar_items=10)
    
def recommend_all(col, w1=1, w2 = 0, w3 = 0, num_similar_items=10):
    if(fe_tech=='Bag of Words'):
        features = fe_using_BoW(st.session_state.df_temp, col)
    elif(fe_tech=='TF-IDF'):
        features = fe_using_TF_IDF(st.session_state.df_temp, col)
    elif(fe_tech=='Word2Vec'):
        features = fe_using_Word2Vec(st.session_state.df_temp, col)
    elif(fe_tech=='Distil BERT'):
        features = fe_using_Distil_BERT(st.session_state.df_temp, col) 
    recom_prods = []
    for i in range(st.session_state.df_temp.shape[0]):
        if(fe_tech not in ['Word2Vec', 'Distil BERT']):
            couple_dist = pairwise_distances(features, features[i])
        else:
            couple_dist = pairwise_distances(features, features[i].reshape(1,-1))
        if(feature2 in st.session_state.df_inp.columns.tolist()):
            feature2_dist = feature_featurization(st.session_state.df_temp, feature2, i) 
        if(feature3 in st.session_state.df_inp.columns.tolist()):
            feature3_dist = feature_featurization(st.session_state.df_temp, feature3, i) 
        if(feature2 in st.session_state.df_inp.columns.tolist() and feature3 in st.session_state.df_inp.columns.tolist()):
            weighted_couple_dist = (float(w1) * couple_dist +  float(w2) * feature2_dist + float(w3) * feature3_dist)/(float(w1) + float(w2) + float(w3))
            indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        elif(feature2 in st.session_state.df_inp.columns.tolist()):
            weighted_couple_dist = (float(w1) * couple_dist +  float(w2) * feature2_dist)/(float(w1) + float(w2))
            indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        elif(feature3 in st.session_state.df_inp.columns.tolist()):
            weighted_couple_dist = (float(w1) * couple_dist +  float(w3) * feature3_dist)/(float(w1) + float(w3))
            indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()
        else:
            weighted_couple_dist = couple_dist
            indices = np.argsort(weighted_couple_dist.ravel())[0:num_similar_items]
        recom_prods.append(st.session_state.df_inp.loc[indices, prod_id].tolist())
    df_recom = st.session_state.df_inp.copy()
    df_recom["Recommended Products"] = recom_prods
    df_recom[f"Recommended Products {title}"] = df_recom["Recommended Products"].apply(lambda x : [st.session_state.prodid_to_title[i] for i in x])
    if(path.exists(f"{root_dir}/{dir}/{fe_tech}/newly_recommended_items.csv")==False):
        #df_recom[[prod_id, title, "Recommended Products"]].to_csv(f"{dir}/{fe_tech}/newly_recommended_items.csv", index = False)
        #newly_recom = pd.read_csv(f"{dir}/{fe_tech}/newly_recommended_items.csv")
        newly_recom = df_recom[[prod_id, title, "Recommended Products", f"Recommended Products {title}"]]
        st.download_button(
           "Download the Recommendations",
           convert_df(newly_recom),
           "newly_recommended_items.csv",
           "text/csv",
           key='download-csv'
           )
# In[ ]:


st.write("""
         ## Content based Product Recommendation
         """)
st.caption("Recommendation App developed by [**Blue Altair's DS Team**](https://www.bluealtair.com/)")
root_dir = "Intermediate_Folder"
dir = "Content_based"
if(os.path.exists(f"{root_dir}/{dir}")==False):
    os.makedirs(f"{root_dir}/{dir}")
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    with st.container():
        fe_tech = st.selectbox('Select Text Featurization Technique',  ('Bag of Words', 'TF-IDF', 'Word2Vec', 'Distil BERT'))


# In[ ]:


def load_data(uploaded_file): 
    filename = uploaded_file.name
    if filename[filename.rfind('.')+1:].lower()=="csv":
        df = pd.read_csv(uploaded_file)
    elif filename[filename.rfind('.')+1:].lower()=="xlsx":
        df = pd.read_excel(uploaded_file)
    else:
        print("Upload a csv or excel file")
    return df

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


# In[ ]:


if uploaded_file:
    df_inp = load_data(uploaded_file)
    w1 = 1
    w2 = 0 
    w3 = 0
    with st.form("form1"):
        prod_id = st.selectbox('Select Product ID Column',  df_inp.columns)
        col11, col12 = st.columns(2)
        with col11:
            title = st.selectbox('Select Title/Feature1 Column',  df_inp.columns, key = 0)
        with col12:
            w1 = st.number_input("Enter the Weight for Feature 1", min_value = 0.0, max_value = 1.0, step = 0.01, value = 1.0, help = 'Select a weight from 0 to 1')
        #hybrid_model = st.expander('Add more features to make a Hybrid Model')
        with st.expander('Add more features to make a Weighted Model'):
            col21, col22 = st.columns(2)
            with col21:
                feature2 = st.selectbox('Select Feature 2',  ['<Select>'] + df_inp.columns.tolist(), key = 2)
            with col22:
                w2 = st.number_input('Enter the Weight for Feature 2', min_value = 0.0, max_value = 1.0,  step = 0.01,help = 'Select a weight from 0 to 1')
            col31, col32 = st.columns(2)
            with col31:
                feature3 = st.selectbox('Select Feature 3',  ['<Select>'] + df_inp.columns.tolist(), key = 3)
            with col32:
                w3 = st.number_input('Enter the Weight for Feature 3', min_value = 0.0, max_value = 1.0,  step = 0.01, help = 'Select a weight from 0 to 1')
        product_id = st.text_input("Enter the Product ID for recommendation demo")
        submitted = st.form_submit_button("Recommend")
    if((w1+w2+w3)==1):
        if submitted:
            if 'df_inp' not in st.session_state:
                st.session_state.df_inp = df_inp
            #if(feature2 in df_inp.columns.tolist() and feature3 in df_inp.columns.tolist()):
             #   if 'df_inp' not in st.session_state:
              #      st.session_state.df_inp = df_inp[[prod_id, title, feature2, feature3]]
            #else:
             #   if 'df_inp' not in st.session_state:
              #      st.session_state.df_inp = df_inp[[prod_id, title]]
            #st.dataframe(df_inp.head())
            st.session_state.df_inp = preprocessing(st.session_state.df_inp, title)
            if 'prodid_to_title' not in st.session_state:
                st.session_state.prodid_to_title = pd.Series(st.session_state.df_inp[title].values,index=st.session_state.df_inp[prod_id]).to_dict()
            if 'df_temp' not in st.session_state:    
                st.session_state.df_temp = st.session_state.df_inp.copy()
            st.session_state.df_temp = text_cleaning(st.session_state.df_temp, title)
            model(product_id, title, w1, w2, w3)
        download_checkbox = st.checkbox("Do you want to download the recommendations for complete data?")
        if download_checkbox:
            recommend_all(title, w1, w2, w3, num_similar_items=10)
        #if(path.exists(f"{dir}/{fe_tech}/newly_recommended_items.csv")==True):
         #   newly_recom = pd.read_csv(f"{dir}/{fe_tech}/newly_recommended_items.csv")
          #  st.download_button(
           #    "Download the Recommendations",
            #   convert_df(newly_recom),
             #  "newly_recommended_items.csv",
              # "text/csv",
              # key='download-csv'
               # )
    else:
        st.warning('Sum of weights should be equal to 1', icon="⚠️")
       
           

