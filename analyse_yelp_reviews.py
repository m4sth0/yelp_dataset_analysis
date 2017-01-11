#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Beispielanalyse der User Reviews des Yelp Dataset Challenge Datensatzes

Die User Reviews für verschiedenen Unternehmen bestehen aus einem Kommentartext 
und einer Bewertung auf einer Skala von 0 bis 5. Es wird die Fragestellung 
untersucht, ob sich die positiven Bewertungen (4-5 Sterne) von den negativen 
(0-2 Sterne) und den neutralen Bewertungen an Hand der Kommentartexte 
unterscheiden lassen und ob die vergebenen Bewertungspunkte direkt aus den 
dazugehörigen Texten abgeleitet werden können.

Mit der Hilfe von Text Mining Algorithmen lassen sich die Kommentare analysieren
und die jeweiligen vergebenen Bewertungspunkte vorhersagen. Diese Sentiment
(Stimmungs)erkennung wird im folgenden mit Hilfe der NLTK-Bibliothek und einer
Naive Baysian Klassifikation umgesetzt.
"""

import json
import matplotlib
import nltk
import numpy as np
import os
import random
import sys
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

# Matplotlib Backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#  NLTK-Package importieren
nltk.download('punkt')

def file_len(fname):
    """ Rückgabe der Dateilänge"""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def create_feats(words):
    """ Konverierung von Wörterlisten in NLTK Feature Sets"""
    return dict([('contains(%s)' % word, True) for word in words])

def reduce_stars(value):
    """ Aggregation der User Bewertungen"""
    if value < 3:
        result = 'bad'
    elif value == 3:
        result = 'neutral'
    else:
        result = 'good'
    
    return(result)

def prep_reviews(rfile, n=None, reduce=True, random=False):
    """ Generator für Unternehmensreview Tokens
 
    Dieser Generator importiert die Reviews aus der angegebenen JSON Datei
    und überführt die Kommentare in ein NLTK kompatibles Format
    Argumente:
        rfile   JSON Review-Datei des Yelp Datensatzes 
        n       Anzahl der importierten Kommentare
        reduce  Aufteilung der Bewertungen in gute (3-5) und schlechte (0-2) 
                Reviews. Default = True
        random  Gleichverteilte zufällige Auswahl. Default = False
    """
    if random:
        nrev = file_len(review_path)
        samples = np.random.choice(np.arange(0, nrev), n)
    counter = 0
    with open(rfile) as filein:
        for line in filein:
            if random and counter not in samples:
                counter += 1
                continue
            elif not random and (counter >= n):
                return
            content = json.loads(line)
            comment = word_tokenize(content['text'])
            star = content['stars']
            if reduce:
                reduce_starts(star)
            else:
                rating = star
            yield ([i.encode('utf-8').lower() for i in comment], rating)
            counter += 1

# Lokale Pfad für die Review Datei
#datapath = '/data/tleppelt/studies/NY'
datapath = './'

review_path = os.path.join(datapath, 'yelp_academic_dataset_review.json')

# Anzahl der Reviews
nrev = file_len(review_path)  # 2685066

# Größe des Subsets
n = 10000

# Random Seed festsetzen
seed = np.random.seed(1001)

# Import und Präprozessierung der Reviews
reviews = prep_reviews(review_path, n, False, True)
data = [sent for sent in reviews]

# Aggregierung der Bewertungen
data_red = [(i[0], reduce_stars(i[1])) for i in data]

# Explorative Datenanalyse
plt = matplotlib.pyplot
t = [i[1] for i in data]
vert_hist = np.histogram(t, bins=5)
height = 0.5
yrange = np.arange(1, 6, 1.)
plt.xlabel('Anzahl n')
plt.ylabel('Bewertung')
plt.title('Yelp Dataset (Subset)- User Reviews')
plt.yticks(yrange)
plt.barh(yrange,vert_hist[0], color=['r', 'r', 'y', 'g', 'g'], height=height, 
         align='center')
#plt.show()
plt.savefig('/tmp/yelp_reviews.png')

# Aufteilung in Training und Testdatensatz
cutoff = int(len(data_red)*9/10)

trainfeats = data_red[:cutoff]
testfeats = data_red[cutoff:]
print('Aufteilung in %d Training und %d Testkommentare' % (len(trainfeats), len(testfeats)))

# Weitere Präprozessierung der Reviews

# Neue Instanz der SentimentAnalyzer Klasse
sentim_analyzer = SentimentAnalyzer()

# Markierung von Negierungen 
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in trainfeats])

# Extraktion von häufigen Wörtern
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
print('Anzahl der Unigram Features: %d' % len(unigram_feats))
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

training_set = sentim_analyzer.apply_features(trainfeats)
test_set = sentim_analyzer.apply_features(testfeats)

# Kalibrierung des Naive-Bayes-Modells mit dem Trainingsdatensatz
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

# Validierung des Modells mit dem Testdatensatz
print('Model Accuracy:', nltk.classify.util.accuracy(classifier, test_set))
classifier.show_most_informative_features()

# Test von zwei selbsterstellten Kommentaren
example_pos = create_feats(word_tokenize("This is a good Restaurant, delicious meals".lower()))
example_neg = create_feats(word_tokenize("It was dirty and horrible. The service was rude and the location in terrible condition".lower()))

print(classifier.classify(example_pos))
print(classifier.classify(example_neg))

