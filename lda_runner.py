#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, TfidfModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

pd.set_option('display.max_colwidth', -1)

import string
from nltk.corpus import stopwords


# ## LDA

# In[8]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[9]:


mallet_path = './mallet-2.0.8/bin/mallet' # update this path


# In[10]:


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(bigram_mod, texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(trigram_mod, texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(
            mallet_path, 
            corpus=corpus, 
            num_topics=num_topics, 
            id2word=dictionary,
            alpha=0.1,
            workers=10,
        )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def get_number_of_topics(coherence_values):
    # Print the coherence scores
    current_m = None
#     for m, cv in enumerate(coherence_values):
#         if cv >= 0.7:
#             current_m = m
#             break
    
#     for m, cv in enumerate(coherence_values):
#         if cv >= 0.6:
#             current_m = m
#             break

#     if current_m == None:
#         for m, cv in enumerate(coherence_values):
#             if cv >= 0.55:
#                 current_m = m
#                 break
                
    if current_m == None:
        for m, cv in enumerate(coherence_values):
            if cv >= 0.5:
                current_m = m
                break

    if current_m == None:
        for m, cv in enumerate(coherence_values):
            if cv >= 0.45:
                current_m = m
                break
    
    return current_m

def get_dominant_topics(optimal_model, corpus, list_contents):
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=list_contents)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    return df_dominant_topic, df_topic_sents_keywords

def get_topic_dominant_doc(df_topic_sents_keywords):
    # Group top 5 sentences under each topic
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0])], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]
    
    return sent_topics_sorteddf_mallet


def get_keyphrase(text):
    # 1. create a TopicRank extractor.
    extractor = pke.unsupervised.TopicRank()

    # 2. load the content of the document.
    extractor.load_document(input=text)

    # 3. select the longest sequences of nouns and adjectives, that do
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)

    # 4. build topics by grouping candidates with HAC (average linkage,
    #    threshold of 1/4 of shared stems). Weight the topics using random
    #    walk, and select the first occuring candidate from each topic.
    extractor.candidate_weighting(threshold=0.74, method='average')

    # 5. get the 10-highest scored candidates as keyphrases
    return extractor.get_n_best(n=1)

def save_topic_words(optimal_model, number_of_topics, id2word, idx):
    res = {}
    for idx in range(number_of_topics):
        res['idx'] = {id2word[item[0]]: item[1] for item in gensim_optimal_model.get_topic_terms(k)}

    json.dump(res, open(f'{week}_topic_words.json', 'w'))
    
def prepare_corpus(list_contents):
    data_words = list(sent_to_words(list_contents))
        
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    #nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        nlp, 
#         data_words_bigrams,
        data_words,
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    )

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    id2word.filter_extremes(no_below=1, no_above=0.5, keep_n=None)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return id2word, corpus, data_lemmatized

def get_topic_names(df_topic_sents_keywords, number_of_topics):
    df_topic_with_dominant_text = get_topic_dominant_doc(df_topic_sents_keywords)

    topic_names = {k: None for k in range(number_of_topics)}
    for k in range(number_of_topics):
        topic_names[k] = get_keyphrase(
            ' '.join(
                df_topic_with_dominant_text[df_topic_with_dominant_text['Topic_Num'] == k].sort_values(
                    'Topic_Perc_Contrib', ascending=False
                ).head(2)['Text'].values)
        )
        
    return topic_names

def compute_lda(list_contents):
    id2word, corpus, data_lemmatized = prepare_corpus(list_contents)

    limit=15; start=2; step=2;
    model_list, coherence_values = compute_coherence_values(
        dictionary=id2word, 
        corpus=corpus, 
        texts=data_lemmatized, 
        start=start, 
        limit=limit, 
        step=step
    )

    topic_options = range(start, limit, step)

    right_topics_index = get_number_of_topics(coherence_values)

    optimal_model = model_list[right_topics_index]
    number_of_topics = topic_options[right_topics_index]

    df_text_with_dominant_topic, df_topic_sents_keywords = get_dominant_topics(
        optimal_model, 
        corpus, 
        list_contents
    )

    topic_names = get_topic_names(df_topic_sents_keywords, number_of_topics)

    return df_text_with_dominant_topic, topic_names


if __name__ == '__main__':

    interesting_country = 'lebanon'
    data = pd.read_csv(interesting_country + '_scraped_data.csv', index_col=0)

    data['better_date'] = data['0'].apply(lambda x: x.split('-')[0])
    data['timestamp'] = pd.to_datetime(data['better_date'])
    data['week'] = data['timestamp'].dt.week

    data['week'].unique()

    data['full_text'] = data['full_text'].apply(lambda x: x.encode('ascii', 'ignore').decode())

    data.sort_values('timestamp', inplace=True)
    data.drop_duplicates('full_text', inplace=True)
    data = data.loc[
      data['full_text'].str[750:1250].drop_duplicates().index
    ]

    all_results = []

    for k in list(data['week'].unique()):
        list_contents = data[data['week'] == k]['full_text'].values.tolist()
        results, topic_names = compute_lda(list_contents)
        results['topic_name'] = results['Dominant_Topic'].apply(lambda x: topic_names[x])
        results['week'] = k
        all_results.append(results)


    concat_all_results = pd.concat(all_results)
    data_with_topics = pd.merge(
        data,
        concat_all_results,
        left_on=['full_text', 'week'],
        right_on=['Text', 'week']
    )

    data_with_topics.to_csv(interesting_country + '_topics_data_friday.csv')