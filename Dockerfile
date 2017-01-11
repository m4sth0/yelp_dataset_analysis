FROM python:2

ADD analyse_yelp_reviews.py / 

ADD yelp_academic_dataset_review.json / 

RUN pip install numpy nltk matplotlib

