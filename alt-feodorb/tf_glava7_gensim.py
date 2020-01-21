# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 18:09:24 2019

@author: feodorb
"""
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import datetime

from gensim.corpora.wikicorpus import WikiCorpus

print('Training: Started at {}'.format(datetime.datetime.now().time()))

wiki=WikiCorpus('K:/ruwiki-20191001-pages-articles-multistream.xml.bz2', dictionary=False)

print('Training: Corpus loaded at {}'.format(datetime.datetime.now().time()))

from gensim.models.phrases import Phrases, Phraser

bigram = Phrases(wiki.get_texts(), min_count=30, progress_per=10000)
bigram_transformer = Phraser(bigram)

print('Training: Bigrams processed at {}'.format(datetime.datetime.now().time()))

def text_generator_bigram():
    for text in wiki.get_texts():
        yield bigram_transformer[[word.decode('utf-8') for word in text]]

trigram = Phrases(text_generator_bigram(), min_count=30, progress_per=10000)
trigram_transformer = Phraser(trigram)

print('Training: Trigrams processed at {}'.format(datetime.datetime.now().time()))

def text_generator_trigram():
    for text in wiki.get_texts():
        yield trigram_transformer[bigram_transformer[[word.decode('utf-8') for word in text]]]

from gensim.models.word2vec import Word2Vec

model = Word2Vec(size=100, window=7, min_count=10, workers = 10)

print('Training: Model created at {}'.format(datetime.datetime.now().time()))

model.build_vocab(text_generator_trigram())

print('Training: Vocab built at {}'.format(datetime.datetime.now().time()))

model.train(text_generator_trigram())

print('Training: trained at {}'.format(datetime.datetime.now().time()))

print(model.most_similar('микеланджело'))

