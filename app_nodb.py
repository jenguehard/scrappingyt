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
# import config
# from func import main, nettoyage, topic_modeling, text_clustering, mysql_connect, insert_user, insert_comment, insert_video, get_data
# from mysql.connector import MySQLConnection, Error
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, silhouette_score
from boto.s3.connection import S3Connection
api_key = S3Connection(os.environ['api_key'], os.environ['api_key'])

YOUTUBE_VIDEO_URL = 'https://www.youtube.com/watch?v={youtube_id}'
YOUTUBE_COMMENTS_AJAX_URL_OLD = 'https://www.youtube.com/comment_ajax'
YOUTUBE_COMMENTS_AJAX_URL_NEW = 'https://www.youtube.com/comment_service_ajax'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'

youtube = build('youtube', 'v3', developerKey=api_key)

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

        st.header('Téléchargement des commentaires de la vidéo !')
        count = 0
        sys.stdout.write('Downloaded %d comment(s)\r' % count)
        sys.stdout.flush()
        start_time = time.time()
        with st.spinner('En cours de téléchargement....'):
            for comment in download_comments(youtube_id):
                output.append(comment)
                count += 1
                sys.stdout.write('Downloaded %d comment(s)\r' % count)
                sys.stdout.flush()

        # st.write(('\n[{:.2f} secondes] Terminé !'.format(time.time() - start_time)))
        st.success("Téléchargement terminé !")
        return output

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)


##############################################################################

def nettoyage(texte, language):
    tex=[]
    # Construction de la liste de stop words
    import stop_words
    sw_1=stop_words.get_stop_words(language)
    from nltk.corpus import stopwords
    sw_nltk = set(stopwords.words(dict_lang[language]))
    sw=list(set(sw_1+list(sw_nltk)))+[str(i) for i in range(100)]+["http", "https", "www"]
   
    texte=texte.lower()
   
    texte=re.sub(r'\W', ' ', texte)
   
    for elem in texte.split():
        if elem in sw or elem==' ':
            continue
        else:
            tex.append(elem)
    return ' '.join(tex)

##############################################################################

def get_dictionnary_of_unique_words(data):
    data["text_clean"] = data["text_clean"].apply(lambda x : x.split(" "))
    unique_words = []

    for i in range(data.shape[0]):
        split_words = data["text_clean"][i]
        unique_words = set(unique_words).union(set(split_words))
    
    unique_words_dict = dict.fromkeys(unique_words, 0)
        
    for i in range(data.shape[0]):
        for word in data["text_clean"][i]:
            unique_words_dict[word] += 1
    sorted_unique_words_dict = {k: v for k, v in sorted(unique_words_dict.items(), key=lambda item: item[1], reverse = True)}
    return unique_words_dict, sorted_unique_words_dict

def get_wordcloud(data_dict):
    wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(data_dict)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.show()
    st.pyplot()

##############################################################################

def topic_modeling(data):
    n_topics = 5
    random_state = 0
    vec = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
    features = vec.fit_transform(data.text_clean)
    cls = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
    feature_names = vec.get_feature_names()
    cls.fit(features)

    # number of most influencing words to display per topic
    n_top_words = 15

    for i, topic_vec in enumerate(cls.components_):
    # topic_vec.argsort() produces a new array
    # in which word_index with the least score is the
    # first array element and word_index with highest
    # score is the last array element. Then using a
    # fancy indexing [-1: -n_top_words-1:-1], we are
    # slicing the array from its end in such a way that
    # top `n_top_words` word_index with highest scores
    # are returned in desceding order
        final_list = []
        final_list.append(str(i))
        for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
            final_list.append(feature_names[fid])
        st.write(" ".join(final_list))
        st.write()

##############################################################################

# Text clustering

def text_clustering(data):
    random_state = 0
    vec = TfidfVectorizer(stop_words="english")
    vec.fit(data.text_clean.values)
    features = vec.transform(data.text_clean.values)

    cls = KMeans(n_clusters=2, random_state=random_state)
    cls.fit(features)


    # predict cluster labels for new dataset
    cls.predict(features)

    # to get cluster labels for the dataset used while
    # training the model (used for models that does not
    # support prediction on new dataset).
    # cls.labels_ - not required here

    # reduce the features to 2D
    pca = PCA(n_components=2, random_state=random_state)
    reduced_features = pca.fit_transform(features.toarray())

    # reduce the cluster centers to 2D
    reduced_cluster_centers = pca.transform(cls.cluster_centers_)

    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    plt.show()
    st.pyplot()

    return cls, features

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
    try :
        req2 = youtube.videos().list(part='snippet,contentDetails', id=youtube_link.split("=")[1])
        res2 = req2.execute()
        language = detect(res2["items"][0]["snippet"]["description"])
    except:
        language = "en"
    
    st.write("Le langage est le suivant " + str(language) + ".")

    if language == "fr" :
        tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
        model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
        
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    
    else:
        classifier = pipeline('sentiment-analysis')
##############################################################################

    # Get all the Youtube comments for the video in "youtube_link".

    data = main(youtube_link)
    df = pd.DataFrame(data)

    # Split the comment from real comment to answers to comment and keep only the real comments.

    df["answer"] = 0
    for i in range(df.shape[0]):
        df["answer"][i] = len(df.cid[i].split("."))-1
    data_clean = df[df.answer == 0][["text", "author", "votes"]].reset_index(drop = True)
    
    data_clean["language"] = 0 
    for i in range(data_clean.shape[0]):
        try :
            data_clean["language"][i] = detect(str(data_clean.text[i]))
        except:
            data_clean["language"][i] = "unknown"

    data_clean = data_clean[data_clean.language == language].reset_index(drop = True)

    # Do sentiment analysis on each comment.
    
    st.header("Processing...")
    my_bar = st.progress(0)
    data_clean["sentiment"] = 0
    for i in range(data_clean.shape[0]):
        my_bar.progress(i/data_clean.shape[0])

        if len(data_clean.text[i]) > 500:
            pass
        else:
            if classifier(data_clean.text[i])[0]["label"] == 'POSITIVE':
                data_clean.sentiment[i] = 'POSITIVE'
            else:
                data_clean.sentiment[i] = 'NEGATIVE'


    # Use the nettoyage function to remove stop words and lower text to create "text_clean" column.
    try :
        data_clean["text_clean"] = data_clean.text.apply(lambda x : nettoyage(x, language))
    
    except :
        data_clean["text_clean"] = data_clean.text.apply(lambda x : nettoyage(x, 'en'))
        print("Language not found")

    data_author = data_clean.author.unique()
    
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

    most_author = data_clean.groupby("author").count().sort_values(by="text", ascending = False).index.to_list()[:5]
    most_comments = data_clean.groupby("author").count().sort_values(by="text", ascending = False)["sentiment"].to_list()[:5]

    
    best_commenters = {}
    for i in range(len(most_author)):
        commenters = {}
        commenters["Author"] = most_author[i]
        commenters["Number"] = most_comments[i]
        st.write("L'utilisateur " + str(most_author[i]) + " a écrit " + str(most_comments[i]) + " commentaire(s).")
        best_commenters[i] = commenters

    # Create two wordclouds and topic modeling : one for positive comments, the other for negative comments.

    data_pos = data_clean[data_clean.sentiment == "POSITIVE"].reset_index(drop = True)
    data_neg = data_clean[data_clean.sentiment == "NEGATIVE"].reset_index(drop = True)

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

    st.header("Récupérez les résultats au format JSON.")

    dict_result = {"link":youtube_link, "description": res2["items"][0]["snippet"]["description"], "key_words" : key_words, "Homogeneity score": homo_score,"Silhouette score" : sil_score, "Most liked comments" : most_liked_comments, "Ratio POSITIVE/NEGATIVE": dd, "Authors" : best_commenters, "Top 30 words full": list(sorted_unique_words_dict.keys())[:30], "Top 30 words positive": list(sorted_unique_words_dict_pos_only.keys())[:30], "Top 30 words negative" : list(sorted_unique_words_dict_neg_only.keys())[:30]}

    st.write()
    if st.button("Téléchargez le rapport"):
        with open("result.json", "w") as outfile:  
            json.dump(dict_result, outfile) 
