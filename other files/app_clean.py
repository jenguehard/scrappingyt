from __future__ import print_function

import io
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import sys
import time
import streamlit as st
import unidecode
import re
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import argparse
import lxml.html
import requests
from lxml.cssselect import CSSSelector
from transformers import pipeline
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from langdetect import detect
import config
import pymysql
from func import main, nettoyage, topic_modeling, text_clustering, mysql_connect, insert_user, insert_comment, insert_video, get_data, get_dictionnary_of_unique_words, get_wordcloud
# from mysql.connector import MySQLConnection, Error
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, silhouette_score


# API SETUP

api_key= config.api_key
youtube = build('youtube', 'v3', developerKey=api_key)


YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL_OLD = 'https://www.youtube.com/comment_ajax'
YOUTUBE_COMMENTS_AJAX_URL_NEW = 'https://www.youtube.com/comment_service_ajax'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'

##############################################################################
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

dict_lang = {'hu':'hungarian',
 'sw':'swedish',
 'no':'norwegian',
 'fi':'finnish',
 'ar':'arabic',
 'id':'indonesian',
 'pt':'portuguese',
 'tr':'turkish',
 'sl':'slovene',
 'es':'spanish',
 'da':'danish',
 'ne':'nepali',
 'ro':'romanian',
 'gr':'greek',
 'nl':'dutch',
 'de':'german',
 'en':'english',
 'ru':'russian',
 'fr':'french',
 'it':'italian'}

local_css("style.css")

st.title("Que pensent les utilisateurs de cette vidéo ?")

st.header("Entrez le lien de votre vidéo.")

youtube_link = st.text_input("Votre lien Youtube")

if youtube_link:
    st.video(youtube_link)

    mydb, mycursor = mysql_connect()

    if mydb.open:
        db_Info = mydb.get_server_info()
        print("Connected to MySQL database... MySQL Server version on ",db_Info)

    data_clean = get_data(youtube_link)
    # Get video's description from Youtube API.
    
    st.header("Description de la vidéo :")
    
    req2 = youtube.videos().list(part='snippet,contentDetails', id=youtube_link.split("=")[1])
    res2 = req2.execute()
    st.write(res2["items"][0]["snippet"]["description"])

    try:
        key_words = " ".join(res2["items"][0]["snippet"]["tags"])
    except :
        key_words = "No tags"

    # Creation of "label" based on sentiment analysis of comments.

    data_clean["label"] =0
    for i in range(data_clean.shape[0]):
        if data_clean.sentiment[i] == "POSITIVE":
            data_clean.label[i] = 1

    # Create dictionnary of unique words and create wordcloud from it.

    unique_words_dict, sorted_unique_words_dict = get_dictionnary_of_unique_words(data_clean)

    data_clean["text_clean"] = data_clean["text_clean"].apply(lambda x : " ".join(x))

    st.header("Quels sont les mots qui ressortent le plus des commentaires ?")

    get_wordcloud(unique_words_dict)
    
    # Topic modeling on comments to check the main subjects.
    try:
        st.header("Quels sont les principaux sujets ?")

        topic_modeling(data_clean)

    except:
        pass

    # Text clustering on comments to check if vocabulary is really different between "POSITIVE" and "NEGATIVE" comments.
    try:
        model, feat = text_clustering(data_clean)

        homo_score = homogeneity_score(data_clean.label, model.predict(feat))
        st.write("Le score d'homogénéité est de " + str(homo_score) + ". Selon la documentation, le score varie entre 0 et 1 où 1 signifie un labelling parfaitement homogène.")
        
        sil_score = silhouette_score(feat, labels=model.predict(feat))
        st.write("Le silhouette score est de " + str(sil_score) + ".La meilleure valeur est 1 et la pire valeur est -1. Les valeurs proches de 0 indiquent des clusters qui se chevauchent. Les valeurs négatives indiquent généralement qu'un échantillon a été attribué au mauvais cluster, car un cluster différent est plus similaire.")

    except:
        pass
    # Count the number of "POSITIVE" and "NEGATIVE" comments.

    st.header("Quel est le sentiment général des commentaires ?")

    data_clean = data_clean[data_clean.sentiment != 0]
    dd = data_clean.sentiment.value_counts().to_dict()

    plt.bar(dd.keys(), dd.values())
    st.pyplot()

    # Get five most liked comments and their sentiment.

    st.header("Quels sont les commentaires les plus 'likés' et leur sentiment ?")

    data_clean.votes = data_clean.votes.astype("int")
    best = data_clean.sort_values(by="votes", ascending = False).index.to_list()[:5]
    most_liked_comments = {}
    for i in best:
        st.write(data_clean.text[i])
        most_liked_comments[i] = data_clean.text[i]
        if data_clean.sentiment[i] == "POSITIVE" :
            st.markdown("**POSITIVE**")
        else:
            st.markdown("**NEGATIVE**")
        st.write("-------------------")

    # Get the five users that commented the most.

    st.header("Quels sont les utilisateurs qui ont le plus commenté ?")

    most_author = data_clean.groupby("user_id").count().sort_values(by="text", ascending = False).index.to_list()[:5]
    most_comments = data_clean.groupby("user_id").count().sort_values(by="text", ascending = False)["sentiment"].to_list()[:5]
    authors = []

    for author in most_author:
        query = """SELECT * from users WHERE user_id= %s"""
        arg = (author,)
        mycursor.execute(query, arg)
        result = mycursor.fetchall()
        authors.append(result[0][1])
    
    best_commenters = {}
    for i in range(len(authors)):
        commenters = {}
        commenters["Author"] = authors[i]
        commenters["Number"] = most_comments[i]
        st.write("L'utilisateur " + str(authors[i]) + " a écrit " + str(most_comments[i]) + " commentaire(s).")
        best_commenters[i] = commenters

    # Create two wordclouds and topic modeling : one for positive comments, the other for negative comments.

    data_pos = data_clean[data_clean.sentiment == "POSITIVE"].reset_index()
    data_neg = data_clean[data_clean.sentiment == "NEGATIVE"].reset_index()

    unique_words_dict_pos, sorted_unique_words_dict_pos = get_dictionnary_of_unique_words(data_pos)
    unique_words_dict_neg, sorted_unique_words_dict_neg = get_dictionnary_of_unique_words(data_neg)


    unique_words_dict_pos_only = {x:unique_words_dict_pos[x] for x in unique_words_dict_pos if x not in unique_words_dict_neg}
    sorted_unique_words_dict_pos_only = {k: v for k, v in sorted(unique_words_dict_pos_only.items(), key=lambda item: item[1], reverse = True)}
    unique_words_dict_neg_only = {x:unique_words_dict_neg[x] for x in unique_words_dict_neg if x not in unique_words_dict_pos}
    sorted_unique_words_dict_neg_only = {k: v for k, v in sorted(unique_words_dict_neg_only.items(), key=lambda item: item[1], reverse = True)}


    data_neg["text_clean"] = data_neg["text_clean"].apply(lambda x : " ".join(x))
    data_pos["text_clean"] = data_pos["text_clean"].apply(lambda x : " ".join(x))

    for word_dict, data in zip([unique_words_dict_pos_only, unique_words_dict_neg_only], [data_pos, data_neg]):        
        if word_dict == unique_words_dict_pos_only:
            st.header("Quels sont les mots qui ressortent le plus des commentaires positifs ?")
        else :
            st.header("Quels sont les mots qui ressortent le plus des commentaires négatifs ?")
        get_wordcloud(word_dict)
        st.header("Quels sont les principaux sujets ?")
        topic_modeling(data)

    if(mydb.open):
        mycursor.close()
        mydb.close()
        print("MySQL connection is closed")


    st.header("Récupérez les résultats au format JSON.")

    dict_result = {"link":youtube_link, "description": res2["items"][0]["snippet"]["description"], "key_words" : key_words, "Homogeneity score": homo_score,"Silhouette score" : sil_score, "Most liked comments" : most_liked_comments, "Ratio POSITIVE/NEGATIVE": dd, "Authors" : best_commenters, "Top 30 words full": list(sorted_unique_words_dict.keys())[:30], "Top 30 words positive": list(sorted_unique_words_dict_pos_only.keys())[:30], "Top 30 words negative" : list(sorted_unique_words_dict_neg_only.keys())[:30]}

    st.write()
    if st.button("Téléchargez le rapport"):
        with open("result.json", "w") as outfile:  
            json.dump(dict_result, outfile) 
