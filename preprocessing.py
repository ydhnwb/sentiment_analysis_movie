import pandas as pd
import re
import string
from string import punctuation
from nltk.corpus import stopwords
class Preprocessing :
    def __init__(self):
        print("Initializing preprocessing...")
        pass

    def clean_text(self, text):
        words = text.split()
        for word in words:
            if str(word).__contains__('@'):
                print("removing "+word)
                text = text.replace(word, '')
            if str(word).__contains__('http'):
                text = text.replace(word, '')
            if str(word).__contains__("#"):
                text = text.replace(word, '')
        text = str(text).strip()
        if len(text) != 0:
            return text
        return ''

    def clean_csv(self, csv_file):
        data = pd.read_csv(csv_file)
        data = data.dropna(axis= 0, how='any')
        data['text'] = data['text'].apply(self.clean_text)
        data_cleaned = pd.DataFrame(data= {'text': data['text']})
        data_cleaned.to_csv("data_cleaned.csv", header= False, index= True, encoding= "utf-8")

    def processTweet(tweet):
        tweet = re.sub(r'\&\w*;', '', tweet)
        tweet = re.sub('@[^\s]+','',tweet)
        tweet = re.sub(r'\$\w*', '', tweet)
        tweet = tweet.lower()
        tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
        tweet = re.sub(r'#\w*', '', tweet)
        tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        tweet = re.sub(r'\s\s+', ' ', tweet)
        tweet = tweet.lstrip(' ')
        tweet = ''.join(c for c in tweet if c <= '\uFFFF')
        return tweet

    def text_process(raw_text):
        nopunc = [char for char in list(raw_text) if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words('english')]



