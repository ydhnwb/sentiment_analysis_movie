import pandas as pd
import numpy
import json
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
from sklearn.externals import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB

imdb_dataset = pd.read_csv("imdb_labelled.txt", sep="\t", header=None)
imdb_dataset.columns = ['text', 'label']
positives = imdb_dataset['label'][imdb_dataset.label == 1]
negatives = imdb_dataset['label'][imdb_dataset.label == 0]
COLNAMES = ["id", "text", "location", "coordinates", "full_name", "country", "country_code", "place_type"]
nltk.download('stopwords')


def word_count(text):
    return len(str(text).split())


imdb_dataset["word_count"] = imdb_dataset["text"].apply(word_count)
print(imdb_dataset)

all_words = []
for line in list(imdb_dataset['text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

hasil_crawl = pd.read_csv("hasil_crawl.csv", names=COLNAMES)
print(hasil_crawl)


def geo_mean(x):
    y = json.loads(x)
    y = numpy.asarray(y)
    mean_geoloc = numpy.add(numpy.add((y[:][0][0]), (y[:][0][1])) / 2, (numpy.add((y[:][0][2]), (y[:][0][3])) / 2)) / 2
    return numpy.around(mean_geoloc, decimals=6)


hasil_crawl['geo_code'] = hasil_crawl['coordinates'].apply(geo_mean)
hasil_crawl = hasil_crawl.drop(['coordinates', 'place_type'], axis=1)
print(hasil_crawl)

####

hasil_crawl.to_pickle("hasil_crawl.p")
hasil_crawl_pickle = pd.read_pickle("hasil_crawl.p")
hasil_crawl_pickle['text'] = hasil_crawl_pickle['text'].apply(Preprocessing.processTweet)
hasil_crawl_pickle = hasil_crawl_pickle.drop_duplicates('text')
hasil_crawl_pickle.shape

eng_stop_words = stopwords.words('english')
hasil_crawl_pickle = hasil_crawl_pickle.copy()
hasil_crawl_pickle['tokens'] = hasil_crawl_pickle['text'].apply(Preprocessing.text_process)
print(hasil_crawl_pickle['tokens'])

# V
bow_transformer = CountVectorizer(analyzer=Preprocessing.text_process).fit(hasil_crawl_pickle['text'])
messages_bow = bow_transformer.transform(hasil_crawl_pickle['text'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)

# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(imdb_dataset['text'][:1000], imdb_dataset['label'][:1000],
                                                    test_size=0.2)

pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii', stop_words='english', lowercase=True)),
    # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

parameters = {'bow__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3), }

# do 10-fold cross validation for each of the 6 possible combinations of the above params
grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
grid.fit(X_train, y_train)

# summarize results
print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
print('\n')
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

joblib.dump(grid, "model.pkl")
# test model
model_NB = joblib.load("model.pkl")

# get predictions from best model above
y_preds = model_NB.predict(X_test)

print('accuracy score: ', str(accuracy_score(y_test, y_preds)*100) + "%")
print('confusion matrix: \n', confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))

# testing
model_NB = joblib.load("model.pkl")

#value random dari database imdb_labelled
sample_str = "It had some average acting from the main person, and it was a low budget as you clearly can see."



def label_to_str(x):
    if x == 0:
        return 'Negative'
    else:
        return 'Positive'

x = 0
text_ = [0] * len(imdb_dataset)
label_ = [0] * len(imdb_dataset)

for review in imdb_dataset['text']:
    predict = model_NB.predict([review])
    text_[x] = review
    label_[x] = label_to_str(predict[0])
    x += 1
    # print("the sentence: \n\n'{}' \n\nhas a {} sentiment".format(review, sentiment_str(p[0])))

print("write ke csv")
hehe = {"text": text_, "label": label_}
hehe2 = pd.DataFrame(data= hehe)
hehe2.to_csv('test_ulang_dataset.csv', header=True, index=False, encoding='utf-8')
hasil_test_ulang = pd.read_csv("test_ulang_dataset.csv", header='infer')
hasil_test_ulang.columns = ['text', 'label']

recheck_pos = hasil_test_ulang['label'][hasil_test_ulang.label == "Positive"]
recheck_neg = hasil_test_ulang['label'][hasil_test_ulang.label == "Negative"]

print("Hasil test ulang punya positive prediksi sebanyak : "+ str(len(recheck_pos)) +" dan negatif sebanyak " + str(len(recheck_neg)))
