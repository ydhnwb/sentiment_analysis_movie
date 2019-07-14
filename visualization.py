import joblib
import pandas as pd
import csv
import matplotlib.pyplot as plt


model = joblib.load("model.pkl")

def label_to_str(x):
    if x == 0:
        return 'Negatif'
    else:
        return 'Positif'

print("test data from twitter")
from_twitter = pd.read_csv("darkphoenix.csv")
from_twitter.columns = ['id', 'tweet']
csvFile = open('movies_predicted.csv', 'w', encoding='utf-8')
csvWriter = csv.writer(csvFile)
for tweet in list(from_twitter['tweet']):
    h = model.predict([tweet])
    csvWriter.writerow([str(tweet), label_to_str(h[0])])

from_twitter_predicted = pd.read_csv("movies_predicted.csv")
from_twitter_predicted.columns = ['tweet', 'label']
labels = 'Positive', 'Negative'
sizes = [len(from_twitter_predicted[from_twitter_predicted['label'] == "Positif"]), len(from_twitter_predicted[from_twitter_predicted['label'] == "Negatif"])]
colors = ['lightgreen', 'red']

# Plot
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()
