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
import config
from mysql.connector import MySQLConnection, Error
from googleapiclient.discovery import build

# API SETUP

api_key= config.api_key
youtube = build('youtube', 'v3', developerKey=api_key)

YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL_OLD = 'https://www.youtube.com/comment_ajax'
YOUTUBE_COMMENTS_AJAX_URL_NEW = 'https://www.youtube.com/comment_service_ajax'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'


def find_value(html, key, num_chars=2, separator='"'):
    pos_begin = html.find(key) + len(key) + num_chars
    pos_end = html.find(separator, pos_begin)
    return html[pos_begin: pos_end]


def ajax_request(session, url, params=None, data=None, headers=None, retries=5, sleep=20):
    for _ in range(retries):
        response = session.post(url, params=params, data=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        if response.status_code in [403, 413]:
            return {}
        else:
            time.sleep(sleep)


def download_comments(youtube_id, sleep=.1):
    if r'\"isLiveContent\":true' in requests.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id)).text:
        print('Live stream detected! Not all comments may be downloaded.')
        return download_comments_new_api(youtube_id, sleep)
    return download_comments_old_api(youtube_id, sleep)


def download_comments_new_api(youtube_id, sleep=1):
    # Use the new youtube API to download some comments
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))
    html = response.text
    session_token = find_value(html, 'XSRF_TOKEN', 3)

    data = json.loads(find_value(html, 'window["ytInitialData"] = ', 0, '\n').rstrip(';'))
    for renderer in search_dict(data, 'itemSectionRenderer'):
        ncd = next(search_dict(renderer, 'nextContinuationData'), None)
        if ncd:
            break
    continuations = [(ncd['continuation'], ncd['clickTrackingParams'])]

    while continuations:
        continuation, itct = continuations.pop()
        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_NEW,
                                params={'action_get_comments': 1,
                                        'pbj': 1,
                                        'ctoken': continuation,
                                        'continuation': continuation,
                                        'itct': itct},
                                data={'session_token': session_token},
                                headers={'X-YouTube-Client-Name': '1',
                                         'X-YouTube-Client-Version': '2.20200207.03.01'})

        if not response:
            break
        if list(search_dict(response, 'externalErrorMessage')):
            raise RuntimeError('Error returned from server: ' + next(search_dict(response, 'externalErrorMessage')))

        # Ordering matters. The newest continuations should go first.
        continuations = [(ncd['continuation'], ncd['clickTrackingParams'])
                         for ncd in search_dict(response, 'nextContinuationData')] + continuations

        for comment in search_dict(response, 'commentRenderer'):
            yield {'cid': comment['commentId'],
                   'text': ''.join([c['text'] for c in comment['contentText']['runs']]),
                   'time': comment['publishedTimeText']['runs'][0]['text'],
                   'author': comment.get('authorText', {}).get('simpleText', ''),
                   'channel': comment['authorEndpoint']['browseEndpoint']['browseId'],
                   'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                   'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
                   'heart': next(search_dict(comment, 'isHearted'), False)}

        time.sleep(sleep)


def search_dict(partial, key):
    if isinstance(partial, dict):
        for k, v in partial.items():
            if k == key:
                yield v
            else:
                for o in search_dict(v, key):
                    yield o
    elif isinstance(partial, list):
        for i in partial:
            for o in search_dict(i, key):
                yield o


def download_comments_old_api(youtube_id, sleep=1):
    # Use the old youtube API to download all comments (does not work for live streams)
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT

    # Get Youtube page with initial comments
    response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))
    html = response.text

    reply_cids = extract_reply_cids(html)

    ret_cids = []
    for comment in extract_comments(html):
        ret_cids.append(comment['cid'])
        yield comment

    page_token = find_value(html, 'data-token')
    session_token = find_value(html, 'XSRF_TOKEN', 3)

    first_iteration = True

    # Get remaining comments (the same as pressing the 'Show more' button)
    while page_token:
        data = {'video_id': youtube_id,
                'session_token': session_token}

        params = {'action_load_comments': 1,
                  'order_by_time': True,
                  'filter': youtube_id}

        if first_iteration:
            params['order_menu'] = True
        else:
            data['page_token'] = page_token

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_OLD, params, data)
        if not response:
            break

        page_token, html = response.get('page_token', None), response['html_content']

        reply_cids += extract_reply_cids(html)
        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment

        first_iteration = False
        time.sleep(sleep)

    # Get replies (the same as pressing the 'View all X replies' link)
    for cid in reply_cids:
        data = {'comment_id': cid,
                'video_id': youtube_id,
                'can_reply': 1,
                'session_token': session_token}

        params = {'action_load_replies': 1,
                  'order_by_time': True,
                  'filter': youtube_id,
                  'tab': 'inbox'}

        response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_OLD, params, data)
        if not response:
            break

        html = response['html_content']

        for comment in extract_comments(html):
            if comment['cid'] not in ret_cids:
                ret_cids.append(comment['cid'])
                yield comment
        time.sleep(sleep)


def extract_comments(html):
    tree = lxml.html.fromstring(html)
    item_sel = CSSSelector('.comment-item')
    text_sel = CSSSelector('.comment-text-content')
    time_sel = CSSSelector('.time')
    author_sel = CSSSelector('.user-name')
    vote_sel = CSSSelector('.like-count.off')
    photo_sel = CSSSelector('.user-photo')
    heart_sel = CSSSelector('.creator-heart-background-hearted')

    for item in item_sel(tree):
        yield {'cid': item.get('data-cid'),
               'text': text_sel(item)[0].text_content(),
               'time': time_sel(item)[0].text_content().strip(),
               'author': author_sel(item)[0].text_content(),
               'channel': item[0].get('href').replace('/channel/','').strip(),
               'votes': vote_sel(item)[0].text_content() if len(vote_sel(item)) > 0 else 0,
               'photo': photo_sel(item)[0].get('src'),
               'heart': bool(heart_sel(item))}


def extract_reply_cids(html):
    tree = lxml.html.fromstring(html)
    sel = CSSSelector('.comment-replies-header > .load-comments')
    return [i.get('data-cid') for i in sel(tree)]


def main(link):
    try:
        youtube_id = link.split("=")[1]
        output = []

        if not youtube_id:
            raise ValueError('you need to specify a Youtube ID and an output filename')

        if os.sep in output:
            outdir = os.path.dirname(output)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        st.write('Downloading Youtube comments for video !')
        count = 0
        sys.stdout.write('Downloaded %d comment(s)\r' % count)
        sys.stdout.flush()
        start_time = time.time()
        for comment in download_comments(youtube_id):
            output.append(comment)
            count += 1
            sys.stdout.write('Downloaded %d comment(s)\r' % count)
            sys.stdout.flush()

        st.write(('\n[{:.2f} secondes] Terminé !'.format(time.time() - start_time)))
        return output

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)

##############################################################################

def nettoyage(texte):
    tex=[]
    # Construction de la liste de stop words
    import stop_words
    sw_1=stop_words.get_stop_words('en')
    from nltk.corpus import stopwords
    sw_nltk = set(stopwords.words('english'))
    sw=list(set(sw_1+list(sw_nltk)))+[str(i) for i in range(100)]
   
    texte=texte.lower()
   
    texte=re.sub(r'\W', ' ', texte)
   
    for elem in texte.split():
        if elem in sw or elem==' ':
            continue
        else:
            tex.append(elem)
    return ' '.join(tex)

##############################################################################

# MySQL connection

import mysql.connector

mydb = mysql.connector.connect(
host="localhost",
user="jenguehard",
password="Testsimplon_99",
database="youtube"
)

mycursor = mydb.cursor(buffered=True)

if mydb.is_connected():
    db_Info = mydb.get_server_info()
    print("Connected to MySQL database... MySQL Server version on ",db_Info)

def insert_video(video_id, link, key_words):
    query = "INSERT INTO videos(video_id, url, key_words) " \
            "VALUES(%s,%s,%s)"
    args = (video_id, link, key_words)

    try:
        mycursor.execute(query, args)
        mydb.commit()
    except Error as error:
        print(error)

def insert_user(user_id, full_name):
    query = "INSERT INTO users(user_id, full_name) " \
            "VALUES(%s,%s)"
    args = (user_id, str(full_name))

    try:
        mycursor.execute(query, args)
        mydb.commit()
    except Error as error:
        print(error)

def insert_comment(comment_id, comment, user_id, video_id, comment_clean, sentiment, votes):
    query = """ INSERT INTO comments(comment_id, comment, user_id, video_id, comment_clean, sentiment, votes) VALUES (%s,%s,%s,%s,%s,%s,%s)"""
    
    args = (comment_id, comment, user_id, video_id, comment_clean, sentiment, votes)
    
    try:
        mycursor.execute(query, args)
        mydb.commit()
    except Error as error:
        print(error)

##############################################################################

# Table creation

mycursor.execute("""
CREATE TABLE IF NOT EXISTS `videos`
(
 `video_id`  int NOT NULL ,
 `url`       text NOT NULL ,
 `key_words`       text NOT NULL ,

PRIMARY KEY (`video_id`)
);

""")

mycursor.execute("""
CREATE TABLE IF NOT EXISTS `users`
(
 `user_id`   int NOT NULL ,
 `full_name` VARCHAR(512) NOT NULL ,

PRIMARY KEY (`user_id`)
);
""")

mycursor.execute("""
CREATE TABLE IF NOT EXISTS comments
(
 comment_id    int NOT NULL DEFAULT 0,
 comment      VARCHAR(512) NOT NULL ,
 user_id       int NOT NULL DEFAULT 0,
 video_id      int NOT NULL DEFAULT 0,
 comment_clean VARCHAR(512) NOT NULL ,
 sentiment     text NOT NULL,
 votes      int NOT NULL DEFAULT 0,

PRIMARY KEY (comment_id),

FOREIGN KEY (user_id) REFERENCES users(user_id),
FOREIGN KEY (video_id) REFERENCES videos(video_id)
);
""")

##############################################################################

st.title("Que pensent les utilisateurs de cette vidéo ?")

st.header("Entrez le lien de votre vidéo.")

youtube_link = st.text_input("Votre lien Youtube")

classifier = pipeline('sentiment-analysis')

if youtube_link:
    st.video(youtube_link)

##############################################################################

    # Check if the data already exist on the database

    query_vid = """SELECT EXISTS(SELECT * from videos WHERE url=%s)"""
    arg_vid = (youtube_link,)
    try:
        mycursor.execute(query_vid, arg_vid)

    except Error as error:
        print(error)

    result_vid = mycursor.fetchall()

    # If it exists retrieve all the information from the database

    if result_vid[0][0] == 1:
        query = """SELECT * from videos WHERE url=%s"""
        arg = (youtube_link,)
        mycursor.execute(query, arg)
        result = mycursor.fetchall()
        video_id = result[0][0]

        query = """SELECT * from comments WHERE video_id = %s"""
        arg = (video_id,)
        mycursor.execute(query, arg)
        result = mycursor.fetchall()

        data_clean = pd.DataFrame(result, columns =['id', 'text', 'user_id', 'video_id', 'text_clean', 'sentiment', 'votes'])

    # Otherwise retrieve all the information from the video and insert it in the database.

    else :
        
        # Get all the Youtube comments for the video in "youtube_link".

        data = main(youtube_link)
        df = pd.DataFrame(data)

        # Split the comment from real comment to answers to comment and keep only the real comments.

        df["answer"] = 0
        for i in range(df.shape[0]):
            df["answer"][i] = len(df.cid[i].split("."))-1
        data_clean = df[df.answer == 0][["text", "author", "votes"]].reset_index()
        
        # Do sentiment analysis on each comment.
        
        data_clean["sentiment"] = 0
        for i in range(data_clean.shape[0]):
            if len(data_clean.text[i]) > 512:
                pass
            else:
                if classifier(data_clean.text[i])[0]["label"] == 'POSITIVE':
                    data_clean.sentiment[i] = 'POSITIVE'
                else:
                    data_clean.sentiment[i] = 'NEGATIVE'

        # Use the nettoyage function to remove stop words and lower text to create "text_clean" column.

        data_clean["text_clean"] = data_clean.text.apply(lambda x : nettoyage(x))
        
        data_author = data_clean.author.unique()

        req2 = youtube.videos().list(part='snippet,contentDetails', id=youtube_link.split("=")[1])
        res2 = req2.execute()

        key_words = " ".join(res2["items"][0]["snippet"]["tags"])

        # Put the video url and key words on the table videos

        mycursor.execute("""SELECT COUNT(*) from videos""")
        result = mycursor.fetchall()
        num_rows = result[0][0]
        video_id = num_rows +1

        insert_video(video_id, youtube_link, key_words)

        # Check if an author is already in table ```users```

        mycursor.execute("""SELECT COUNT(*) from users""")
        result = mycursor.fetchall()
        num_rows = result[0][0]

        for i in range(len(data_author)):

            query = """SELECT EXISTS(SELECT * from users WHERE full_name=%s)"""
            arg = (data_author[i],)
            mycursor.execute(query, arg)

            result = mycursor.fetchall()

            if result[0][0] == 1:
                continue
            else:
                num_rows += 1
                insert_user(num_rows, data_author[i])

        mycursor.execute("""SELECT COUNT(*) from comments""")
        result = mycursor.fetchall()
        num_rows = result[0][0]

        for i in range(data_clean.shape[0]):
            ### Get user id

            num_rows += 1

            id_comment = num_rows

            query_user = """SELECT * from users WHERE full_name = %s"""
            string_user  = str(data_clean.author[i])

            arg = (string_user,)

            mycursor.execute(query_user, arg)

            myresult_user = mycursor.fetchall()
            user_id = myresult_user[0][0]
            
            insert_comment(id_comment, str(data_clean.text[i]), user_id, video_id, data_clean.text_clean[i], data_clean.sentiment[i], data_clean.votes[i])
        
        query = """SELECT * from comments WHERE video_id = %s"""
        arg = (video_id,)
        mycursor.execute(query, arg)
        result = mycursor.fetchall()
        
        data_clean = pd.DataFrame(result, columns =['id', 'text', 'user_id', 'video_id', 'text_clean', 'sentiment', 'votes'])

    # Get video's description from Youtube API.
    
    req2 = youtube.videos().list(part='snippet,contentDetails', id=youtube_link.split("=")[1])
    res2 = req2.execute()
    st.write(res2["items"][0]["snippet"]["description"])

    # Create dictionnary of unique words and create wordcloud from it.

    unique_words = []

    for i in range(data_clean.shape[0]):
        split_words = data_clean["text_clean"][i].split(" ")
        unique_words = set(unique_words).union(set(split_words))
        
    unique_words_dict = dict.fromkeys(unique_words, 0)
    data_clean["text_clean"] = data_clean["text_clean"].apply(lambda x : x.split(" "))

    for i in range(data_clean.shape[0]):
        for word in data_clean["text_clean"][i]:
            unique_words_dict[word] += 1

    data_clean["text_clean"] = data_clean["text_clean"].apply(lambda x : " ".join(x))

    st.header("Description de la vidéo :")
    
    req2 = youtube.videos().list(part='snippet,contentDetails', id=youtube_link.split("=")[1])
    res2 = req2.execute()
    st.write(res2["items"][0]["snippet"]["description"])


    st.header("Quels sont les mots qui ressortent le plus des commentaires ?")

    wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(unique_words_dict)
        
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    st.pyplot()

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

    for i in best:
        st.write(data_clean.text[i])
        if data_clean.sentiment[i] == "POSITIVE" :
            st.markdown("**POSITIVE**")
        else:
            st.markdown("**NEGATIVE**")
        st.write("-------------------")

    # Get the five users that commented the most.

    st.header("Quels sont les utilisateurs qui ont le plus commenter ?")

    most_author = data_clean.groupby("user_id").count().sort_values(by="text", ascending = False).index.to_list()[:5]
    most_comments = data_clean.groupby("user_id").count().sort_values(by="text", ascending = False)["sentiment"].to_list()[:5]
    authors = []

    for author in most_author:
        query = """SELECT * from users WHERE user_id= %s"""
        arg = (author,)
        mycursor.execute(query, arg)
        result = mycursor.fetchall()
        authors.append(result[0][1])

    for i in range(len(authors)):
        st.write("L'utilisateur " + str(authors[i]) + " a écrit " + str(most_comments[i]) + " commentaire(s).")

    # Create two wordclouds : one for positive comments, the other for negative comments.

    data_pos = data_clean[data_clean.sentiment == "POSITIVE"].reset_index()
    data_pos["text_clean"] = data_pos["text_clean"].apply(lambda x : x.split(" "))
    unique_words_pos = []

    for i in range(data_pos.shape[0]):
        split_words_pos = data_pos["text_clean"][i]
        unique_words_pos = set(unique_words_pos).union(set(split_words_pos))

    unique_words_dict_pos = dict.fromkeys(unique_words_pos, 0)

    for i in range(data_pos.shape[0]):
        for word in data_pos["text_clean"][i]:
            unique_words_dict_pos[word] += 1

    data_neg = data_clean[data_clean.sentiment == "NEGATIVE"].reset_index()
    data_neg["text_clean"] = data_neg["text_clean"].apply(lambda x : x.split(" "))
    unique_words_neg = []

    for i in range(data_neg.shape[0]):
        split_words_neg = data_neg["text_clean"][i]
        unique_words_neg = set(unique_words_neg).union(set(split_words_neg))
    
    unique_words_dict_neg = dict.fromkeys(unique_words_neg, 0)
        
    for i in range(data_neg.shape[0]):
        for word in data_neg["text_clean"][i]:
            unique_words_dict_neg[word] += 1

    unique_words_dict_pos_only = {x:unique_words_dict_pos[x] for x in unique_words_dict_pos if x not in unique_words_dict_neg}
    unique_words_dict_neg_only = {x:unique_words_dict_neg[x] for x in unique_words_dict_neg if x not in unique_words_dict_pos}

    for word_dict in [unique_words_dict_pos_only, unique_words_dict_neg_only]:        
        if word_dict == unique_words_dict_pos_only:
            st.header("Quels sont les mots qui ressortent le plus des commentaires positifs ?")
        else :
            st.header("Quels sont les mots qui ressortent le plus des commentaires négatifs ?")

        wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(word_dict)
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud)
        plt.show()
        st.pyplot()

    if(mydb.is_connected()):
        mycursor.close()
        mydb.close()
        print("MySQL connection is closed")