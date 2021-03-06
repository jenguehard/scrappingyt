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
    if r'"isLiveContent":true' in requests.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id)).text:
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
    session_token = bytes(session_token, 'ascii').decode('unicode-escape')

    data = json.loads(find_value(html, 'var ytInitialData = ', 0, '};') + '}')
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
                                         'X-YouTube-Client-Version': '2.20201202.06.01'})

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
    session_token = bytes(session_token, 'ascii').decode('unicode-escape')

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

# def find_value(html, key, num_chars=2, separator='"'):
#     pos_begin = html.find(key) + len(key) + num_chars
#     pos_end = html.find(separator, pos_begin)
#     return html[pos_begin: pos_end]


# def ajax_request(session, url, params=None, data=None, headers=None, retries=5, sleep=20):
#     for _ in range(retries):
#         response = session.post(url, params=params, data=data, headers=headers)
#         if response.status_code == 200:
#             return response.json()
#         if response.status_code in [403, 413]:
#             return {}
#         else:
#             time.sleep(sleep)


# def download_comments(youtube_id, sleep=.1):
#     if r'\"isLiveContent\":true' in requests.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id)).text:
#         print('Live stream detected! Not all comments may be downloaded.')
#         return download_comments_new_api(youtube_id, sleep)
#     return download_comments_old_api(youtube_id, sleep)


# def download_comments_new_api(youtube_id, sleep=1):
#     # Use the new youtube API to download some comments
#     session = requests.Session()
#     session.headers['User-Agent'] = USER_AGENT

#     response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))
#     html = response.text
#     print(html)
#     session_token = find_value(html, 'XSRF_TOKEN', 3)

#     data = json.loads(find_value(html, 'window["ytInitialData"] = ', 0, '\n').rstrip(';'))
#     for renderer in search_dict(data, 'itemSectionRenderer'):
#         ncd = next(search_dict(renderer, 'nextContinuationData'), None)
#         if ncd:
#             break
#     continuations = [(ncd['continuation'], ncd['clickTrackingParams'])]

#     while continuations:
#         continuation, itct = continuations.pop()
#         response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_NEW,
#                                 params={'action_get_comments': 1,
#                                         'pbj': 1,
#                                         'ctoken': continuation,
#                                         'continuation': continuation,
#                                         'itct': itct},
#                                 data={'session_token': session_token},
#                                 headers={'X-YouTube-Client-Name': '1',
#                                          'X-YouTube-Client-Version': '2.20200207.03.01'})

#         if not response:
#             break
#         if list(search_dict(response, 'externalErrorMessage')):
#             raise RuntimeError('Error returned from server: ' + next(search_dict(response, 'externalErrorMessage')))

#         # Ordering matters. The newest continuations should go first.
#         continuations = [(ncd['continuation'], ncd['clickTrackingParams'])
#                          for ncd in search_dict(response, 'nextContinuationData')] + continuations

#         for comment in search_dict(response, 'commentRenderer'):
#             yield {'cid': comment['commentId'],
#                    'text': ''.join([c['text'] for c in comment['contentText']['runs']]),
#                    'time': comment['publishedTimeText']['runs'][0]['text'],
#                    'author': comment.get('authorText', {}).get('simpleText', ''),
#                    'channel': comment['authorEndpoint']['browseEndpoint']['browseId'],
#                    'votes': comment.get('voteCount', {}).get('simpleText', '0'),
#                    'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
#                    'heart': next(search_dict(comment, 'isHearted'), False)}

#         time.sleep(sleep)


# def search_dict(partial, key):
#     if isinstance(partial, dict):
#         for k, v in partial.items():
#             if k == key:
#                 yield v
#             else:
#                 for o in search_dict(v, key):
#                     yield o
#     elif isinstance(partial, list):
#         for i in partial:
#             for o in search_dict(i, key):
#                 yield o


# def download_comments_old_api(youtube_id, sleep=1):
#     # Use the old youtube API to download all comments (does not work for live streams)
#     session = requests.Session()
#     session.headers['User-Agent'] = USER_AGENT

#     # Get Youtube page with initial comments
#     response = session.get(YOUTUBE_VIDEO_URL.format(youtube_id=youtube_id))
#     html = response.text
#     reply_cids = extract_reply_cids(html)
#     print(reply_cids)
#     ret_cids = []
#     for comment in extract_comments(html):
#         ret_cids.append(comment['cid'])
#         yield comment

#     page_token = find_value(html, 'data-token')
#     session_token = find_value(html, 'XSRF_TOKEN', 3)

#     first_iteration = True

#     # Get remaining comments (the same as pressing the 'Show more' button)
#     while page_token:
#         data = {'video_id': youtube_id,
#                 'session_token': session_token}

#         params = {'action_load_comments': 1,
#                   'order_by_time': True,
#                   'filter': youtube_id}

#         if first_iteration:
#             params['order_menu'] = True
#         else:
#             data['page_token'] = page_token

#         response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_OLD, params, data)
#         if not response:
#             break

#         page_token, html = response.get('page_token', None), response['html_content']

#         reply_cids += extract_reply_cids(html)
#         for comment in extract_comments(html):
#             if comment['cid'] not in ret_cids:
#                 ret_cids.append(comment['cid'])
#                 yield comment

#         first_iteration = False
#         time.sleep(sleep)

#     # Get replies (the same as pressing the 'View all X replies' link)
#     for cid in reply_cids:
#         data = {'comment_id': cid,
#                 'video_id': youtube_id,
#                 'can_reply': 1,
#                 'session_token': session_token}

#         params = {'action_load_replies': 1,
#                   'order_by_time': True,
#                   'filter': youtube_id,
#                   'tab': 'inbox'}

#         response = ajax_request(session, YOUTUBE_COMMENTS_AJAX_URL_OLD, params, data)
#         if not response:
#             break

#         html = response['html_content']

#         for comment in extract_comments(html):
#             if comment['cid'] not in ret_cids:
#                 ret_cids.append(comment['cid'])
#                 yield comment
#         time.sleep(sleep)


# def extract_comments(html):
#     print(html)
#     tree = lxml.html.fromstring(html)
#     print(tree)
#     item_sel = CSSSelector('.comment-item')
#     print(item_sel)
#     text_sel = CSSSelector('.comment-text-content')
#     time_sel = CSSSelector('.time')
#     author_sel = CSSSelector('.user-name')
#     vote_sel = CSSSelector('.like-count.off')
#     photo_sel = CSSSelector('.user-photo')
#     heart_sel = CSSSelector('.creator-heart-background-hearted')

#     for item in item_sel(tree):
#         print(item)
#         yield {'cid': item.get('data-cid'),
#                'text': text_sel(item)[0].text_content(),
#                'time': time_sel(item)[0].text_content().strip(),
#                'author': author_sel(item)[0].text_content(),
#                'channel': item[0].get('href').replace('/channel/','').strip(),
#                'votes': vote_sel(item)[0].text_content() if len(vote_sel(item)) > 0 else 0,
#                'photo': photo_sel(item)[0].get('src'),
#                'heart': bool(heart_sel(item))}
#         print({'cid': item.get('data-cid'),
#                'text': text_sel(item)[0].text_content(),
#                'time': time_sel(item)[0].text_content().strip(),
#                'author': author_sel(item)[0].text_content(),
#                'channel': item[0].get('href').replace('/channel/','').strip(),
#                'votes': vote_sel(item)[0].text_content() if len(vote_sel(item)) > 0 else 0,
#                'photo': photo_sel(item)[0].get('src'),
#                'heart': bool(heart_sel(item))})


# def extract_reply_cids(html):
#     tree = lxml.html.fromstring(html)
#     sel = CSSSelector('.comment-replies-header > .load-comments')
#     return [i.get('data-cid') for i in sel(tree)]


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

        toolbar_width = 40

        # setup toolbar
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

        for i in range(toolbar_width):
            time.sleep(0.1) # do real work here
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("]\n") # this ends the progress bar

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

##############################################################################

st.title("Que pensent les utilisateurs de cette vidéo ?")

st.header("Entrez le lien de votre vidéo.")

youtube_link = st.text_input("Votre lien Youtube")

classifier = pipeline('sentiment-analysis')

if youtube_link:
    st.video(youtube_link)
    data = main(youtube_link)
    df = pd.DataFrame(data)

    df["answer"] = 0
    for i in range(df.shape[0]):
        df["answer"][i] = len(df.cid[i].split("."))-1
    data_clean = df[df.answer == 0][["text", "author"]].reset_index()
    data_clean["sentiment"] = 0
    for i in range(data_clean.shape[0]):
        if len(data_clean.text[i]) > 512:
            pass
        else:
            if classifier(data_clean.text[i])[0]["label"] == 'POSITIVE':
                data_clean.sentiment[i] = 'POSITIVE'
            else:
                data_clean.sentiment[i] = 'NEGATIVE'

    data_clean["text_clean"] = data_clean.text.apply(lambda x : nettoyage(x))

    data_author = data_clean.author.unique()

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

    st.header("Quels sont les mots qui ressortent le plus des commentaires ?")

    wordcloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(unique_words_dict)
        
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    st.pyplot()

    st.header("Quel est le sentiment général des commentaires ?")

    data_clean = data_clean[data_clean.sentiment != 0]
    dd = data_clean.sentiment.value_counts().to_dict()

#    author_list = data_clean.author.value_counts()

    plt.bar(dd.keys(), dd.values())
    st.pyplot()

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