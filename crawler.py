import tweepy
import time
from twitter import Twitter
from preprocessing import Preprocessing
import csv
API = Twitter().instance()
waitQuery = 100
waitTime = 2.0
engineBlow = 1
Preprocessing = Preprocessing()
csvFile = open('darkphoenix.csv', 'w', encoding='utf-8')
csvWriter = csv.writer(csvFile)

def search() :
    global API, waitQuery, waitTime, engineBlow
    query = str(input("Search something : "))
    total_number = int(input("n : "))
    cursor = tweepy.Cursor(API.search, query + " -RT", tweet_mode = "extended", lang = "en").items()
    count = 0
    error = 0
    secondcount = 0
    while secondcount < total_number:
        try:
            c = next(cursor)
            count += 1
            if count % waitQuery == 0:
                time.sleep(waitTime)
        except tweepy.TweepError:
            print("Sleeping...")
            time.sleep(60 * engineBlow)
            c = next(cursor)
        except StopIteration:
            break

        try:
            text_val = c._json['full_text']
            text_val = str(text_val).lower()
            text_val = Preprocessing.processTweet(text_val)
            if "rt" not in text_val:
                if len(text_val) != 0:
                    secondcount += 1
                    csvWriter.writerow([secondcount,str(text_val)])
                    print("[INFO] Getting a tweet : " + str(secondcount) + " = " + text_val)
        except Exception as e:
            error += 1
            print('[EXCEPTION] Stream data: ' + str(e))

search()