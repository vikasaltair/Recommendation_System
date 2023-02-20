#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from io import StringIO
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from surprise import Dataset 
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from surprise import KNNBasic
from surprise import SVD
from surprise import NMF
from surprise import dump
from pyspark import SparkContext, SQLContext   # required for dealing with dataframes
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS      # for Matrix Factorization using ALS 
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import findspark
import pyspark
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation, Dot, Add
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
import sklearn.model_selection
#from sklearn.model_selection import train_test_split
import math
import os
import shutil
from os import path
from itertools import chain
import re
from functools import reduce


# In[ ]:





# In[ ]:


def preprocessing(df, col):
    df = df.dropna() # excluding the records with missing values
    df = df[df[col]<=df[col].quantile(0.90)] # removing outliers
    df = df.reset_index(drop=True)
    return df


# In[ ]:



# In[ ]:



# In[ ]:


def classification_report_surprise(predictions, customer_averages):
    actual_labels = []
    predicted_labels = []
    for cid, pid, quant, est, details in predictions:
        if(details['was_impossible']==False):
            if((quant - customer_averages[cid])>0):
                actual_labels.append("Yes")
            else:
                actual_labels.append("No")
            if((est - customer_averages[cid])>0):
                predicted_labels.append("Yes")
            else:
                predicted_labels.append("No")
    return pd.DataFrame(classification_report(actual_labels, predicted_labels, output_dict=True)).transpose().iloc[:2, :3]
    
def highlight_recommended_items(x, recommended_items, column):
    return ['background-color: yellow' if i in recommended_items else '' for i in x.Items_bought]
    
# In[ ]:
def modelling_surprise(surprise_df, train_df, test_df, anti_df):
    global userid_input
    if(is_numeric_dtype(df[userid])):
        userid_input = int(userid_input)
    if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")==True):
        newly_recom = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")
        train_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy",allow_pickle=True)
        test_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy",allow_pickle=True)
        train_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv")
        test_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv")
        if algo in ['User-user Similarity', 'Item-item Similarity']:
            model_full = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")[0]
        bought_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Actuals {title}'].tolist()[0])
        recommended_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Newly Recommended {title}'].tolist()[0])
    else:
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}")
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")==False):
            model = algo_model[algo]
            model.fit(train_df)
            dump.dump(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model", model)
        else:
            model = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")[0]
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions")==False):
            train_predictions = model.test(train_df.build_testset())
            dump.dump(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions", train_predictions)
        else:
            train_predictions = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions")[0]
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions")==False):
            test_predictions = model.test(test_df)
            dump.dump(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions", test_predictions)
        else:
            test_predictions = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions")[0]
        custIds = []
        prodIds = []
        quants = []
        for cid, pid, quant, est, _ in train_predictions:
            custIds.append(cid)
            prodIds.append(pid)
            quants.append(quant)
        dict = {userid: custIds, itemid: prodIds, rating: quants}
        training_df = pd.DataFrame(dict)
        customer_averages = training_df.groupby(userid)[rating].mean()
        train_rmse = accuracy.rmse(train_predictions, verbose = False)
        np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy", train_rmse)
        test_rmse = accuracy.rmse(test_predictions, verbose = False)
        np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy", train_rmse)
        train_classification_report = classification_report_surprise(train_predictions, customer_averages) 
        train_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv", index = False)
        test_classification_report = classification_report_surprise(test_predictions, customer_averages) 
        test_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv", index = False)
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")==False):
            model_full = algo_model[algo]
            model_full.fit(surprise_df.build_full_trainset())
            dump.dump(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full", model)
        else:
            model_full = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")[0]
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions")==False):
            predictions = model_full.test(anti_df)
            dump.dump(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions", predictions)
        else:
            predictions = dump.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions")[0]
        predictions_df = pd.DataFrame(predictions)
        indices = predictions_df.groupby('uid')['est'].nlargest(5).reset_index()["level_1"].values.tolist()
        predictions_df = predictions_df.iloc[indices,:]
        actuals = df.groupby(userid).agg({itemid: lambda x: list(x)}).reset_index()
        actuals.columns = [userid, "Actuals"]
        newly_recom = predictions_df.groupby("uid").agg({"iid": lambda x: list(x)}).reset_index()
        newly_recom.columns = [userid, "Newly Recommended"]
        newly_recom = pd.merge(newly_recom, actuals, on = userid)
        newly_recom[f"Actuals {title}"] = newly_recom["Actuals"].apply(lambda x : [itemid_to_title[i] for i in x])
        newly_recom[f"Newly Recommended {title}"] = newly_recom["Newly Recommended"].apply(lambda x : [itemid_to_title[i] for i in x])
        newly_recom[[userid, "Actuals", f"Actuals {title}", "Newly Recommended", f"Newly Recommended {title}"]].to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv", index = False)
        bought_items = newly_recom[newly_recom[userid]==userid_input][f'Actuals {title}'].tolist()[0]
        recommended_items = newly_recom[newly_recom[userid]==userid_input][f'Newly Recommended {title}'].tolist()[0]
    #if(is_numeric_dtype(df[userid])):
     #   userid_input = int(userid_input)
    #bought_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Actuals {title}'].tolist()[0])
    #recommended_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Newly Recommended {title}'].tolist()[0])
    st.write(f"Items bought by :blue[{userid_input}] : {bought_items}")
    st.write(f":green[Newly Recommended items] for :blue[{userid_input}] : {recommended_items}")
    if algo in ['User-user Similarity', 'Item-item Similarity']:
        st.markdown(f"Items bought by similar users of :blue[{userid_input}]")
        trainset_full = surprise_df.build_full_trainset()
        nearset_neighbors = [trainset_full.to_raw_uid(i) for i in model_full.get_neighbors(trainset_full.to_inner_uid(userid_input), k = factors)]
        nearset_neighbors_df = df[df[userid].isin(nearset_neighbors)][[userid, title]]
        #nearset_neighbors_df = nearset_neighbors_df[~nearset_neighbors_df[title].isin(bought_items)]
        #nearset_neighbors_df = nearset_neighbors_df[nearset_neighbors_df[title].isin(recommended_items)]
        nearset_neighbors_df = nearset_neighbors_df[(nearset_neighbors_df[title].isin(recommended_items)) | (nearset_neighbors_df[title].isin(bought_items))]
        nearset_neighbors_df.drop_duplicates(inplace = True)
        nearset_neighbors_df = nearset_neighbors_df.groupby(userid).agg({title: lambda x: list(x)}).reset_index()
        nearset_neighbors_df.columns = ['Similar_Users', 'Items_bought']
        patrn = re.compile(r'\b(' + '|'.join(map(str, recommended_items)) + r')\b')
        #pat1 = re.compile(r'\b(' + '|'.join(map(str, bought_items)) + r')\b')
        #replacements = [pat.sub(r'<font color="#66BB55">\1</font>'), pat.sub(r'<font color="#66BB55">\1</font>')]
        #df_styled = nearset_neighbors_df.style.format(lambda txt : pat.sub(r'<font color="#66BB55">\1</font>', repr(txt)))
        df_styled = nearset_neighbors_df.style.format(lambda txt : eval(patrn.sub(r'<font color="#66BB55">\1</font>', repr(re.compile(r'\b(' + '|'.join(map(str, bought_items)) + r')\b').sub(r'<font color="#3F00FF">\1</font>', repr(txt))))))
        st.markdown(df_styled.to_html(table_uuid="table_1"), unsafe_allow_html=True)
        #st.table(nearset_neighbors_df.style.format(lambda txt : pat.sub(r'<font color="#66BB55">\1</font>', repr(txt))))
        #st.dataframe(nearset_neighbors_df.style.apply(highlight_recommended_items, recommended_items, 'Items_bought'))
        #st.dataframe(nearset_neighbors_df.groupby(userid).agg({title: lambda x: list(x)}).reset_index())
    return train_rmse, test_rmse, train_classification_report, test_classification_report  



#### ML models for recommendation from Surprise library
def fitting_surprise(df):
    evaluation_measures = pd.DataFrame({'RMSE': []})
    reader = Reader()
    surprise_df = Dataset.load_from_df(df, reader)
    train_df, test_df = train_test_split(surprise_df, test_size=.20, random_state = 42)
    #cus_avgs = binary_classification(train_df)
    if(path.exists(f"{root_dir}/{dir}/anti_df.csv")==False):    
        anti_df = surprise_df.build_full_trainset().build_anti_testset()
        pd.DataFrame(anti_df, columns = [userid, itemid, rating]).to_csv(f"{root_dir}/{dir}/anti_df.csv", index = False)
    else:
        anti_df = list(pd.read_csv(f"{root_dir}/{dir}/anti_df.csv").itertuples(index=False, name=None))
    train_rmse, test_rmse, train_classification_report, test_classification_report  = modelling_surprise(surprise_df, train_df, test_df, anti_df)
    evaluation_measures.loc['Training Data'] = [train_rmse]#, train_precision, train_recall, train_f1]
    evaluation_measures.loc['Test Data'] = [test_rmse]
    return evaluation_measures, train_classification_report, test_classification_report


# In[ ]:

def new_k_all(list_actual, list_recommended, k=5):
        new_list = set(list_recommended).difference(set(list_actual))
        new_list = [v for v in new_list][:k]
        return new_list
        #return ", ".join(new_list)

def classification_report_als(predictions, customer_averages):
    actual_labels = []
    predicted_labels = []
    pred_itr = predictions.collect()
    for row in pred_itr:
        if(math.isnan(row['prediction'])==False):
            if((row[rating] - customer_averages[row[userid]])>0):
                actual_labels.append("Yes")
            else:
                actual_labels.append("No")
            if((row['prediction'] - customer_averages[row[userid]])>0):
                predicted_labels.append("Yes")
            else:
                predicted_labels.append("No")
    return pd.DataFrame(classification_report(actual_labels, predicted_labels, output_dict=True)).transpose().iloc[:2, :3]



def fitting_als(df):
    global userid_input
    index_to_itemid = {}
    #itemid_to_title = df.select(itemid, title).distinct().rdd.collectAsMap()
    evaluation_measures = pd.DataFrame({'RMSE': []})
    if(is_numeric_dtype(df[userid])):
        userid_input = int(userid_input)
    if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")==True):
        recommendations_all = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")
        train_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy",allow_pickle=True)
        test_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy",allow_pickle=True)
        train_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv")
        test_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv")
        bought_items = eval(recommendations_all[recommendations_all[userid]==userid_input][f'Actuals_{title}'].tolist()[0])
        recommended_items = eval(recommendations_all[recommendations_all[userid]==userid_input][f'Newly_Recommended_{title}'].tolist()[0])
    else:
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}")
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")
        #max_freq = int(df[userid].value_counts().values[0])
        max_freq = df.groupby(userid)['StockCode'].nunique().max()
        df_spark = spark.createDataFrame(df)
        categorical_columns = [item[0] for item in df_spark.dtypes if item[1].startswith('string')]
        if(len(categorical_columns)>0):
            indexer = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
            pipeline = Pipeline(stages=indexer)
            df_spark = pipeline.fit(df_spark).transform(df_spark)
        X_train, X_test = df_spark.randomSplit([0.8,0.2]) 
        user_col = [userid+'_index' if userid+'_index' in df_spark.columns else userid][0]
        item_col = [itemid+'_index' if itemid+'_index' in df_spark.columns else itemid][0]
        #if(type(userid_input)==int):
        if('index' in user_col):
            userid_to_index = df_spark.select(userid, userid+'_index').distinct().rdd.collectAsMap()
            index_to_userid = df_spark.select(userid+'_index', userid).distinct().rdd.collectAsMap()
        if('index' in item_col):
            itemid_to_index = df_spark.select(itemid, itemid+'_index').distinct().rdd.collectAsMap()
            index_to_itemid = df_spark.select(itemid+'_index', itemid).distinct().rdd.collectAsMap()
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")==False):
            als = ALS(userCol=user_col,itemCol=item_col,ratingCol=rating,rank=factors, maxIter=10, seed=0)
            model = als.fit(X_train)
            #als.save(spark, f"{dir}/ALS/model")
            #model.write().overwrite().save(f"{dir}/ALS/model")
        else:
            model = als.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")==False):
            als_full = ALS(userCol=user_col,itemCol=item_col,ratingCol=rating,rank=factors, maxIter=10, seed=0)
            model_full = als_full.fit(df_spark)
        else:
            model_full = als_full.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")
        evaluator=RegressionEvaluator(metricName="rmse",labelCol=rating,predictionCol="prediction")
        train_predictions = model.transform(X_train)
        test_predictions = model.transform(X_test).na.drop()
        customer_averages = df_spark.groupBy(userid).avg(rating).rdd.collectAsMap()
        train_rmse = evaluator.evaluate(train_predictions)
        np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy", np.array([train_rmse]))
        test_rmse = evaluator.evaluate(test_predictions)
        np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy", np.array([test_rmse]))
        train_classification_report = classification_report_als(train_predictions, customer_averages)
        train_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv", index = False)
        test_classification_report = classification_report_als(test_predictions, customer_averages) 
        test_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv", index = False)
        recommendations_all = model_full.recommendForUserSubset(df_spark[[user_col]], max_freq+10)
        recommendations_all = recommendations_all.selectExpr('*',f"recommendations.{item_col} as recommend")
        actuals_all = df_spark.groupby(col(userid)).agg(collect_list(col(itemid)).alias("Actuals"))
        if('index' in user_col):
            actuals_all = actuals_all.join(df_spark.selectExpr(userid, user_col), userid, 'left')
        recommendations_all = actuals_all.join(recommendations_all, user_col, 'left')
        recommendations_all = recommendations_all.withColumn('recommend', coalesce('recommend', array().cast("array<integer>")))
        if('index' in item_col):
            custom_udf = udf(lambda x: [index_to_itemid[v] for v in x],ArrayType(StringType()))
            recommendations_all = recommendations_all.withColumn('recommend', custom_udf('recommend'))
        new_k_all_udf = udf(lambda x,y:new_k_all(x,y),ArrayType(StringType()))
        recommendations_all = recommendations_all.withColumn('Newly_Recommended', new_k_all_udf('Actuals', 'recommend'))
        itemid_to_title_udf = udf(lambda x: [itemid_to_title[v] for v in x],ArrayType(StringType()))
        recommendations_all = recommendations_all.withColumn(f"Actuals_{title}", itemid_to_title_udf('Actuals'))
        recommendations_all = recommendations_all.withColumn(f"Newly_Recommended_{title}", itemid_to_title_udf('Newly_Recommended'))
        recommendations_all = recommendations_all.toPandas()
        recommendations_all[[userid, 'Actuals', f"Actuals_{title}", 'Newly_Recommended', f"Newly_Recommended_{title}"]].to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv", index = False)
    bought_items = recommendations_all[recommendations_all[userid]==userid_input][f'Actuals_{title}'].tolist()[0]
    recommended_items = recommendations_all[recommendations_all[userid]==userid_input][f'Newly_Recommended_{title}'].tolist()[0]
    evaluation_measures.loc['Training Data'] = [train_rmse]#, train_precision, train_recall, train_f1]
    evaluation_measures.loc['Test Data'] = [test_rmse]#, test_precision, test_recall, test_f1]
    st.write(f"Items bought by :blue[{userid_input}] : {bought_items}")
    st.write(f":green[Newly Recommended items] for :blue[{userid_input}] : {recommended_items}")
    return evaluation_measures, train_classification_report, test_classification_report
                
def dl_predict(model, df):
    pred = np.empty((df.shape[0], 1))
    try:
        pred = model.predict([df[f'{userid}_index'].values, df[f'{itemid}_index'].values], batch_size=128)
    except:
        pass
    df['prediction'] = pred.ravel()
    return df

def classification_report_dl(predictions, customer_averages):
    actual_labels = []
    predicted_labels = []
    for i, row in predictions.iterrows():
        if(math.isnan(row['prediction'])==False):
            if((row[rating] - customer_averages[row[userid]])>0):
                actual_labels.append("Yes")
            else:
                actual_labels.append("No")
            if((row['prediction'] - customer_averages[row[userid]])>0):
                predicted_labels.append("Yes")
            else:
                predicted_labels.append("No")
    return pd.DataFrame(classification_report(actual_labels, predicted_labels, output_dict=True)).transpose().iloc[:2, :3]


def dl_model_config(N,P):
    K = factors # latent dimensionality
    #mu = df_keras_train[rating].mean()
    #reg = 0. 
    u = Input(shape=(1,))
    p = Input(shape=(1,))
    u_embedding = Embedding(N, K)(u) 
    p_embedding = Embedding(P, K)(p)
    u_bias = Embedding(N, 1)(u) # (N, 1, 1)
    p_bias = Embedding(P, 1)(p) # (N, 1, 1)
    x = Dot(axes=2)([u_embedding, p_embedding]) # (N, 1, 1)
    x = Add()([x, u_bias, p_bias])
    x = Flatten()(x)
    ##### side branch
    u_embedding = Flatten()(u_embedding) # (N, K)
    p_embedding = Flatten()(p_embedding) # (N, K)
    y = Concatenate()([u_embedding, p_embedding]) # (N, 2K)
    y = Dense(400)(y)
    y = Activation('elu')(y)
    # y = Dropout(0.5)(y)
    y = Dense(1)(y)
    ##### merge
    x = Add()([x, y])
    model = Model(inputs=[u, p], outputs=x)
    model.compile(
      loss='mse',
      # optimizer='adam',
      # optimizer=Adam(lr=0.01),
      optimizer=SGD(lr=0.08, momentum=0.9),
      metrics=['mse'],
    )
    return model

def fitting_deep_learning(df):
    global userid_input
    evaluation_measures = pd.DataFrame({'RMSE': []})
    if(is_numeric_dtype(df[userid])):
        userid_input = int(userid_input)
    if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")==True): 
        newly_recom = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")
        train_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy",allow_pickle=True)
        test_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy",allow_pickle=True)
        train_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv")
        test_classification_report = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv")
        bought_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Actuals {title}'].tolist()[0])
        recommended_items = eval(newly_recom[newly_recom[userid]==userid_input][f'Newly Recommended {title}'].tolist()[0])
    else:
        if(path.exists(f"{dir}/anti_df.csv")==False):
            reader = Reader()
            anti_df = Dataset.load_from_df(df, reader).build_full_trainset().build_anti_testset()
            anti_df = pd.DataFrame(anti_df, columns = [userid, itemid, rating])
            anti_df.to_csv(f"{dir}/anti_df.csv", index = False)
        else:
            anti_df = pd.read_csv(f"{dir}/anti_df.csv")
        df_keras = df.copy()
        df_keras[f'{userid}_index'] = df_keras[userid].astype('category').cat.codes
        userid_to_index = dict(zip(df_keras[userid], df_keras[userid].astype('category').cat.codes))
        df_keras[f'{itemid}_index'] = df_keras[itemid].astype('category').cat.codes
        itemid_to_index = dict(zip(df_keras[itemid], df_keras[itemid].astype('category').cat.codes))
        anti_df[f'{userid}_index'] = anti_df[userid].map(userid_to_index).fillna(anti_df[userid])
        anti_df[f'{itemid}_index'] = anti_df[itemid].map(itemid_to_index).fillna(anti_df[itemid])
        N = df_keras[userid].nunique()
        P = df_keras[itemid].nunique()
        df_keras_train, df_keras_test = sklearn.model_selection.train_test_split(df_keras,train_size = 0.8,random_state=3)
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}")
        if(os.path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")==False):
            os.makedirs(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")==False):
            model = dl_model_config(N,P)
            model.fit(x=[df_keras_train[f'{userid}_index'].values, df_keras_train[f'{itemid}_index'].values],
              y=df_keras_train[rating].values,
              epochs=5,
              batch_size=128,
              validation_data=([df_keras_train[f'{userid}_index'].values, df_keras_train[f'{itemid}_index'].values],
              df_keras_train[rating].values))
            model.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model", model)
        else:
            model = keras.models.load_model(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model")
        #train_rmse = math.sqrt(r.history['mse'][-1])
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy")==False):
            train_rmse = np.sqrt(model.evaluate([df_keras_train[f'{userid}_index'].values, df_keras_train[f'{itemid}_index'].values], df_keras_train[rating].values))
            np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy", train_rmse)
        else:
            train_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_rmse.npy")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/test_rmse.npy")==False):
            test_rmse = np.sqrt(model.evaluate([df_keras_test[f'{userid}_index'].values, df_keras_test[f'{itemid}_index'].values], df_keras_test[rating].values))
            np.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy", train_rmse)
        else:
            test_rmse = np.load(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_rmse.npy")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions.csv")==False):
            train_predictions = dl_predict(model, df_keras_train)
            train_predictions.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions.csv", index = False)
        else:
            train_predictions = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_predictions.csv")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions.csv")==False):
            test_predictions = dl_predict(model, df_keras_test)
            test_predictions.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions.csv", index = False)
        else:
            test_predictions = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_predictions.csv")
        customer_averages = train_predictions.groupby(userid)[rating].mean()
        train_classification_report = classification_report_dl(train_predictions, customer_averages) 
        train_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/train_classification_report.csv", index = False)
        test_classification_report = classification_report_dl(test_predictions, customer_averages)
        test_classification_report.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/test_classification_report.csv", index = False)
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")==False):
            model_full = dl_model_config(N,P)
            model_full.fit(
              x=[df_keras[f'{userid}_index'].values, df_keras[f'{itemid}_index'].values],
              y=df_keras[rating].values,
              epochs=5,
              batch_size=128)
            model_full.save(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full", model_full)
        else:
            model_full = keras.models.load_model(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/model_full")
        if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions.csv")==False):
            predictions = dl_predict(model_full, anti_df)
            predictions.to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions.csv", index = False)
        else:
            predictions = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/predictions.csv")
        predictions_df = predictions.copy()
        indices = predictions_df.groupby(userid)['prediction'].nlargest(5).reset_index()["level_1"].values.tolist()
        predictions_df = predictions_df.iloc[indices,:]
        actuals = df.groupby(userid).agg({itemid: lambda x: list(x)}).reset_index()
        actuals.columns = [userid, "Actuals"]
        newly_recom = predictions_df.groupby(userid).agg({itemid: lambda x: list(x)}).reset_index()
        newly_recom.columns = [userid, "Newly Recommended"]
        newly_recom = pd.merge(newly_recom, actuals, on = userid)
        newly_recom[f"Actuals {title}"] = newly_recom["Actuals"].apply(lambda x : [itemid_to_title[i] for i in x])
        newly_recom[f"Newly Recommended {title}"] = newly_recom["Newly Recommended"].apply(lambda x : [itemid_to_title[i] for i in x])
        newly_recom[[userid, "Actuals", f"Actuals {title}", "Newly Recommended", f"Newly Recommended {title}"]].to_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv", index = False)
        bought_items = newly_recom[newly_recom[userid]==userid_input][f'Actuals {title}'].tolist()[0]
        recommended_items = newly_recom[newly_recom[userid]==userid_input][f'Newly Recommended {title}'].tolist()[0]
    st.write(f"Items bought by :blue[{userid_input}] : {bought_items}")
    st.write(f":green[Newly Recommended items] for :blue[{userid_input}] : {recommended_items}")
    evaluation_measures.loc['Training Data'] = [train_rmse[0]]#, train_precision, train_recall, train_f1]
    evaluation_measures.loc['Test Data'] = [test_rmse[0]]#, test_precision, test_recall, test_f1]
    return evaluation_measures, train_classification_report, test_classification_report


   
# In[3]:

st.write("""
         ## Product Recommendation
         """)
st.caption("Recommendation App developed by [**Blue Altair's DS Team**](https://www.bluealtair.com/)")
root_dir = "Intermediate_Folder"
dir = "Collaborative_Filtering"
if(os.path.exists(f"{root_dir}/{dir}")==False):
    os.makedirs(f"{root_dir}/{dir}")
algo_dirs = {"User-user Similarity" : "user_user", "Item-item Similarity" : "item_item", "SVD" : "SVD", "NMF" : "NMF", "ALS" : "ALS", "Deep Learning" : "Deep_Learning", "AutoEncoder" : "AutoEncoder"}
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file")
    algo = st.selectbox('Select Algorithm type',  ('User-user Similarity', 'SVD', 'NMF', 'ALS', 'Deep Learning'))
    with st.container():
        st.markdown("HyperParameter Tuning")
        if algo in ['User-user Similarity', 'Item-item Similarity']:
            factors = st.slider("Number of nearest neighbors", 20, 100, 40, 10, help = "Try out different values to find optimal value of the Hyperparameter")
        else:
            factors = st.slider("Number of latent factors", 5, 100, 10, 5, help = "Try out different values to find optimal value of the Hyperparameter")
#@st.cache(suppress_st_warning=True)
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

if uploaded_file:
    df = load_data(uploaded_file)
    eval_measures = pd.DataFrame()
    with st.form("form1"):
        col11, col12 = st.columns(2)
        with col11:
            userid = st.selectbox('Select User Id Column',  df.columns)
        with col12:
            itemid = st.selectbox('Select Item Id Column',  df.columns)
        col21, col22 = st.columns(2)
        with col21:
            rating = st.selectbox('Select Ratings Column',  df.columns)
        with col22:
            title = st.selectbox('Select Title/Description Column',  df.columns)
        userid_input = st.text_input("Enter the User Id for recommendation demo")
        submitted = st.form_submit_button("Recommend")
    if submitted:
        algo_model = {"User-user Similarity" : KNNBasic(k = factors, random_state = 42,verbose = False), "Item-item Similarity" : KNNBasic(k = factors, user_based = False, random_state = 42), "SVD" : SVD(n_factors = factors, random_state = 42), "NMF" : NMF(n_factors = factors, random_state = 42)}
        df = df[[userid, itemid, rating, title]]
        df[title] = df[title].str.strip()
        df = preprocessing(df, rating)
        itemid_to_title = pd.Series(df[title].values,index=df[itemid]).to_dict()
        if algo in ['User-user Similarity', 'Item-item Similarity', 'SVD', 'NMF']:
            eval_measures, train_classification_report, test_classification_report = fitting_surprise(df[[userid, itemid, rating]])
            #train_classification_report = fitting_surprise(df[[userid, itemid, rating]])[1]
        elif algo=='ALS':
            findspark.init()
            findspark.find()
            sc = SparkContext.getOrCreate()      # instantiating spark context 
            spark = SparkSession(sc) # instantiating Spark Session
            eval_measures, train_classification_report, test_classification_report = fitting_als(df[[userid, itemid, rating]])
        elif algo=='Deep Learning':
            eval_measures, train_classification_report, test_classification_report = fitting_deep_learning(df[[userid, itemid, rating]])
        elif algo=='AutoRec':
            eval_measures = fitting_autorec(df)
        with st.container(): 
            st.markdown("**Evaluation matrix**")
            col31, col32, col33 = st.columns(3)
            with col31:
                st.markdown("RMSE") 
                st.dataframe(eval_measures)
            with col32:
                st.markdown("Train Classification Report") 
                st.dataframe(train_classification_report)
            with col33:
                st.markdown("Test Classification Report") 
                st.dataframe(test_classification_report)
    if(path.exists(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")==True):
        newly_recom = pd.read_csv(f"{root_dir}/{dir}/{algo_dirs[algo]}/{factors}/newly_recommended_items.csv")
        st.download_button(
       "Download the Recommendations made",
       convert_df(newly_recom),
       "newly_recommended_items.csv",
       "text/csv",
       key='download-csv'
        )