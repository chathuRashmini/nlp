# -*- coding: utf-8 -*-
"""
Created July 2017

@author: arw
"""

import pandas as pd
import numpy as np
from normalization import normalize_corpus
from utils import build_feature_matrix

# Load the cleaned movie reviews dataset
dataset = pd.read_csv(r'movie_reviews_short.csv')
# Print the first few data points
print(dataset.head())

# Divide data into training and testing sets
train_data = dataset[:8000]   # set at 25000 for full file
test_data = dataset[8000:]

# Divide the data into the data (review) and the label (sentiment)
train_reviews = np.array(train_data['review'])
train_sentiments = np.array(train_data['sentiment'])
test_reviews = np.array(test_data['review'])
test_sentiments = np.array(test_data['sentiment'])

# Let's first try a sample dataset for experimenting
# sample_docs = [92, 1817, 4626, 4356, 1008, 3155, 2533, 4002]
sample_docs = [1999]
sample_data = [(test_reviews[index],
                test_sentiments[index])
                  for index in sample_docs]

sample_data    

my_reviews = "This movie was quite amazing and terrifying"
my_review = np.array()



# Normalize the data using the normalization.py module
norm_train_reviews = normalize_corpus(train_reviews,
                                      lemmatize=True,
                                      only_text_chars=True)
# Extract the features - which features? Try other features using those available in utils.py                                                                           
vectorizer, train_features = build_feature_matrix(documents=norm_train_reviews,
                                                  feature_type='tfidf',
                                                  ngram_range=(1, 1), 
                                                  min_df=0.0, max_df=1.0)                                      
                                      
                                      

from sklearn.linear_model import SGDClassifier
# Build/train a SVM model (as we did before for text classification)
svm = SGDClassifier(loss='hinge', n_iter=500)
svm.fit(train_features, train_sentiments)



# Normalize the test reviews                        
norm_test_reviews = normalize_corpus(test_reviews,
                                     lemmatize=True,
                                     only_text_chars=True)  

norm_my_review = normalize_corpus(my_review,
                                     lemmatize=True,
                                     only_text_chars=True)

# Extract features from the test reviews                                   
test_features = vectorizer.transform(norm_test_reviews)  

my_features = vectorizer.transform(norm_my_review)       

# Predict sentiment for sample docs from test data
for doc_index in sample_docs:
    print('Review:-')
    print(test_reviews[doc_index])
    print('Actual Labeled Sentiment:', test_sentiments[doc_index])
    doc_features = test_features[doc_index]
    predicted_sentiment = svm.predict(doc_features)[0]
    print('Predicted Sentiment:', predicted_sentiment)
    print()

predict_my_sentiment = svm.predict(my_features)
   
# Predict the sentiment for test dataset movie reviews
predicted_sentiments = svm.predict(test_features)       

# Evaluate model prediction performance
from utils import display_evaluation_metrics, display_confusion_matrix, display_classification_report

# Show performance metrics
display_evaluation_metrics(true_labels=test_sentiments,
                           predicted_labels=predicted_sentiments,
                           positive_class='positive')  

# Show confusion matrix
display_confusion_matrix(true_labels=test_sentiments,
                         predicted_labels=predicted_sentiments,
                         classes=['positive', 'negative'])

# Show detailed per-class classification report
display_classification_report(true_labels=test_sentiments,
                              predicted_labels=predicted_sentiments,
                              classes=['positive', 'negative'])


