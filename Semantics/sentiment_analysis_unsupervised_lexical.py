# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

import pandas as pd
import numpy as np


# Load the cleaned movie reviews dataset
dataset = pd.read_csv(r'movie_reviews.csv')
# Print the first few data points
print(dataset.head())

# Divide data into training and testing sets
train_data = dataset[:25000]
test_data = dataset[25000:]

# Divide the test data into the data (review) and the label (sentiment)
test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])

# Let's first try a sample dataset for experimenting
sample_docs = [92, 5817, 7626, 7356, 1008, 7155, 3533, 8002]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]

sample_data        


# AFINN is a rich sentiment lexicon with values for polarity and intensity
# It even has scores for smileys!
from afinn import Afinn
afn = Afinn(emoticons=True) 
print(afn.score('I really hated the plot of this movie'))

print(afn.score('I really hated the plot of this movie :('))



# We will try the SentiWordNet and VADER lexicons for Sentiment Analysis

# NLTK provides a nice interface to SentiWordNet
import nltk
import html
from nltk.corpus import sentiwordnet as swn

# Get synset for 'good' from sentiwordnet (SWN)
good = list(swn.senti_synsets('good', 'n'))[0]
# Print synset sentiment scores
print('Positive Polarity Score:', good.pos_score())
print('Negative Polarity Score:', good.neg_score())
print('Objective Score:', good.obj_score())

from normalization import normalize_accented_characters

def analyze_sentiment_sentiwordnet_lexicon(review,
                                           verbose=False):
    # Pre-process text
    review = normalize_accented_characters(review)
    review = html.unescape(review)
    # review = strip_html(review) - we need to write a function for this!
    # Tokenize and POS tag text tokens
    text_tokens = nltk.word_tokenize(review)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0
    # Get wordnet synsets based on POS tags
    # Get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        # If senti-synset is found        
        if ss_set:
            # Add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
    
    # Aggregate final scores
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # Display results in a nice (pandas) table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score,
                                         norm_pos_score, norm_neg_score,
                                         norm_final_score]],
                                         columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Objectivity',
                                                                       'Positive', 'Negative', 'Overall']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print(sentiment_frame)
        
    return final_sentiment
            
# Detailed sentiment analysis for sample reviews       
for review, review_sentiment in sample_data:  
    print('Review:')
    print(review)
    print()
    print('Labeled Sentiment:', review_sentiment)   
    print()    
    final_sentiment = analyze_sentiment_sentiwordnet_lexicon(review,
                                                             verbose=True)
    print('-'*60)                       



# Predict sentiment for test movie reviews dataset - Warning: will take some time
sentiwordnet_predictions = [analyze_sentiment_sentiwordnet_lexicon(review)
                            for review in test_reviews]

from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

# Get model performance statistics
print('Performance metrics (SentiWordNet):')
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=sentiwordnet_predictions,
                           positive_class='positive')  
print('\nConfusion Matrix:')                       
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=sentiwordnet_predictions,
                         classes=['positive', 'negative'])
print('\nClassification report:')                 
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=sentiwordnet_predictions,
                              classes=['positive', 'negative'])  
print()
                                                



# Now we use the VADER lexicon for Sentiment Analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment_vader_lexicon(review, 
                                    threshold=0.1,
                                    verbose=False):
    # Pre-process text
    review = normalize_accented_characters(review)
    review = html.unescape(review)
    # review = strip_html(review)
    # Analyze the sentiment for review
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(review)
    # Get aggregate scores and final sentiment
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    if verbose:
        # Display detailed sentiment statistics
        positive = str(round(scores['pos'], 2)*100)+'%'
        final = round(agg_score, 2)
        negative = str(round(scores['neg'], 2)*100)+'%'
        neutral = str(round(scores['neu'], 2)*100)+'%'
        sentiment_frame = pd.DataFrame([[final_sentiment, final, positive,
                                        negative, neutral]],
                                        columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'], 
                                                                      ['Predicted Sentiment', 'Polarity Score',
                                                                       'Positive', 'Negative',
                                                                       'Neutral']], 
                                                              labels=[[0,0,0,0,0],[0,1,2,3,4]]))
        print(sentiment_frame)
    
    return final_sentiment
        
    
    
# Get detailed sentiment statistics
for review, review_sentiment in sample_data:
    print('Review:')
    print(review)
    print()
    print('Labeled Sentiment:', review_sentiment)
    print()   
    final_sentiment = analyze_sentiment_vader_lexicon(review,
                                                        threshold=0.1,
                                                        verbose=True)
    print('-'*60)                      

# Predict sentiment for test movie reviews dataset - Warning: will take some time
vader_predictions = [analyze_sentiment_vader_lexicon(review, threshold=0.1)
                     for review in test_reviews] 

# Get model performance statistics
print('Performance metrics (Vader):')
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=vader_predictions,
                           positive_class='positive')  
print('\nConfusion Matrix:')                        
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=vader_predictions,
                         classes=['positive', 'negative'])
print('\nClassification report:')                       
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=vader_predictions,
                              classes=['positive', 'negative']) 
print()


