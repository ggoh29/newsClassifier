import pandas as pd
import nltk.tokenize as nt
import nltk
nltk.download('averaged_perceptron_tagger')
import re
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from scipy.special import softmax
import math
import copy
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import eli5
from eli5.sklearn import PermutationImportance


source = r"C:/Users/CHRIS/Documents/"
fake = "Fake.csv"
real = "True.csv"

class newsClassifier:

    def __init__(self):
        self.tokens = {"CC": 1, "CD": 1, "DT": 1, "EX": 1, "FW": 1, "IN": 1, "JJ": 1, "JJR": 1, "JJS": 1, "LS": 1, "MD": 1,
                  "NN": 1, "NNS": 1, "NNP": 1, "NNPS": 1, "PDT": 1, "POS": 1, "PRP": 1, "PRP$": 1, "RB": 1, "RBR":1,
                  "RBS": 1, "RP": 1, "TO": 1, "UH": 1, "VB": 1, "VBD": 1, "VBG": 1, "VBN": 1, "VBP": 1, "VBZ": 1,
                  "WDT": 1, "WP": 1, "WP$": 1, "WRB": 1, "START": 1, "END": 1}

        fake_t_body = {i: copy.deepcopy(self.tokens) for i in self.tokens}
        real_t_body = {i: copy.deepcopy(self.tokens)  for i in self.tokens}
        fake_t_title = {i: copy.deepcopy(self.tokens)  for i in self.tokens}
        real_t_title = {i: copy.deepcopy(self.tokens)  for i in self.tokens}
        self.body_dct = {"fake": fake_t_body, "real": real_t_body}
        self.title_dct = {"fake": fake_t_title, "real": real_t_title}
        self.dct = {"body": self.body_dct, "title": self.title_dct}
        fake_body_mem_of_two = {}
        real_body_mem_of_two = {}
        for token1 in self.tokens:
            for token2 in self.tokens:
                if token2 == "START" or token1 == "END":
                    continue
                else:
                    fake_body_mem_of_two[token1 + " " + token2] = copy.deepcopy(self.tokens)
                    real_body_mem_of_two[token1 + " " + token2] = copy.deepcopy(self.tokens)
        self.memTwoDct = {"fake": fake_body_mem_of_two, "real": real_body_mem_of_two}


    def parseDataframe(self, df, type):

        cols = df.columns
        for index, row in df.iterrows():
            title = row[cols[0]]
            title_tokens = self.tokenize(title)
            self.buildMatrix(title_tokens, self.title_dct[type])
            body = row[cols[1]]
            body_tokens = self.tokenize(body)
            self.buildMatrix(body_tokens, self.body_dct[type])
            self.buildMemOfTwoMatrix(body_tokens, self.memTwoDct[type])


    def tokenize(self, text):
        ss = nt.sent_tokenize(text)
        tokenized_sent = [nt.word_tokenize(sent) for sent in ss]
        tokens = [nltk.pos_tag(sent) for sent in tokenized_sent][0]
        tokens.insert(0, ("START", "START"))
        tokens.append(("END", "END"))
        return tokens


    def buildMatrix(self, tokens_lst, matrix):

        for i in range(1, len(tokens_lst)):
            prev = tokens_lst[i-1][1]
            cur = tokens_lst[i][1]
            if prev not in self.tokens or cur not in self.tokens:
                continue
            else:
                matrix[prev][cur] += 1


    def buildMemOfTwoMatrix(self, tokens_lst, matrix):

        for i in range(2, len(tokens_lst)):
            prevprev = tokens_lst[i-2][1]
            prev =  tokens_lst[i-1][1]
            cur = tokens_lst[i][1]
            if prevprev not in self.tokens or cur not in self.tokens or prev not in self.tokens:
                continue
            else:
                matrix[prevprev + " " + prev][cur] += 1


    def smoothAll(self):

        for txt_type in ['real', 'fake']:
            self.smooth(self.title_dct[txt_type])
            self.smooth(self.body_dct[txt_type])
            self.smooth(self.memTwoDct[txt_type])


    def smooth(self, matrix):
        for row in matrix:
            acc = 0
            for col in matrix[row]:
                acc += matrix[row][col]
            for col in matrix[row]:
                matrix[row][col] = np.log(matrix[row][col]/ acc)


    def generateVector(self, df):
        vec = []

        cols = df.columns
        for index, row in df.iterrows():
            title = row[cols[0]]
            body = row[cols[1]]
            vec.append(self.getProbVector(title, body))

        return vec


    def getProbVector(self, title, body):
        title_score = analyser.polarity_scores(title)["compound"]
        body_score = analyser.polarity_scores(body)["compound"]
        title, body = self.tokenize(title), self.tokenize(body)
        title_t_fake_prob = self.getlogProbTransitionProbMemOfOne(title, 'fake', 'title')
        title_t_real_prob = self.getlogProbTransitionProbMemOfOne(title, 'real', 'title')
        body_t_fake_prob = self.getlogProbTransitionProbMemOfOne(body, 'fake', 'body')
        body_t_real_prob = self.getlogProbTransitionProbMemOfOne(body, 'real', 'body')
        body_t_fake_prob_memTwo = self.getlogProbTransitionProbMemOfTwo(title, 'fake')
        body_t_real_prob_memTwo = self.getlogProbTransitionProbMemOfTwo(title, 'real')
        title_prob = softmax([title_t_fake_prob, title_t_real_prob])
        body_prob = softmax([body_t_fake_prob, body_t_real_prob])
        body_prob_memTwo = softmax([body_t_fake_prob_memTwo, body_t_real_prob_memTwo])
        probVector = np.concatenate((title_prob, body_prob, body_prob_memTwo))
        probVector = list(probVector)
        probVector.append(title_score)
        probVector.append(body_score)
        return probVector


    # return logP(x_0 -> x_1 .... | txt_type)
    def getlogProbTransitionProbMemOfOne(self, token_list, txt_type, place):
        # get transition matrix
        matrix = self.dct[place][txt_type]

        # get the probabilities
        log_prob = 0
        for i in range(1, len(token_list)):
            from_token = token_list[i-1][1]
            to_token = token_list[i][1]
            if from_token not in self.tokens or to_token not in self.tokens:
                continue
            else:
                log_prob += matrix[from_token][to_token]

        return log_prob


    def getlogProbTransitionProbMemOfTwo(self, tokens_lst, type):

        matrix = self.memTwoDct[type]

        acc = 0
        for i in range(2, len(tokens_lst)):
            prevprev = tokens_lst[i - 2][1]
            prev = tokens_lst[i - 1][1]
            cur = tokens_lst[i][1]
            if prevprev not in self.tokens or cur not in self.tokens or prev not in self.tokens:
                continue
            else:
                acc += matrix[prevprev + " " + prev][cur]

        return acc


    

"""using https://www.kaggle.com/c/fake-news/data?select=train.csv as dataset"""
alt = "train.csv"

alt = pd.read_csv(source + alt).sort_values(by=['label']).dropna().reset_index()
alt = alt.drop(columns = ['id', 'index'])
number_of_labels = list(alt['label'].value_counts().values)

fake = alt.iloc[:number_of_labels[1]].reset_index().drop(columns = ['index'])
real = alt.iloc[number_of_labels[1]:].reset_index().drop(columns = ['index'])

bucket_train = 700
bucket_test = 70

"""using https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset as dataset"""
# fake = pd.read_csv(source+fake)
# real = pd.read_csv(source+real)
#
# bucket_train = 1800
# bucket_test = 200

fake_train = fake.head(bucket_train * 10)
real_train = real.head(bucket_train * 10)

y_train = np.array(([0] * bucket_train + [1] * bucket_train) * 10)
x_train = []

fake_train_10 = np.array_split(fake_train, 10)
real_train_10 = np.array_split(real_train, 10)

for i in range(10):
    classifier = newsClassifier()
    for j in range(10):
        if i != j:
            classifier.parseDataframe(fake_train_10[j], "fake")
            classifier.parseDataframe(real_train_10[j], "real")
    classifier.smoothAll()
    vec = classifier.generateVector(pd.concat([fake_train_10[i], real_train_10[i]]))
    for vector in vec:
        x_train.append(vector)


"""hyperparameter tuning - to be commented out once saved"""

# p_test = {'learning_rate':[0.01, 0.05, 0.1, 0.25, 0.5], 'n_estimators':[100, 250, 500, 750, 1000]}
#
# tuning = RandomizedSearchCV(estimator=GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10),
#                             param_distributions = p_test, scoring='accuracy', n_jobs=-1, n_iter=10, cv=5)
#
# tuning.fit(np.asarray(x_train), y_train)
#
# lr = tuning.best_params_['learning_rate']
# ne = tuning.best_params_['n_estimators']
#
# p_test2 = {'max_depth':[2, 3, 4, 5, 6, 7]}
#
# tuning = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=lr, n_estimators=ne, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10),
#             param_grid = p_test2, scoring='accuracy',n_jobs=4, cv=5)
# tuning.fit(np.asarray(x_train), y_train)
#
# md = tuning.best_params_['max_depth']
#
# print("params are: learning rate:" + str(lr) + ",n_estimators:" + str(ne) + ",max_depth:" + str(md))
# "params are: learning rate:0.01,n_estimators:250,max_depth:3"


"""parameter tuned using the above commented code"""
model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=250, max_depth=3, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10)
model.fit(np.asarray(x_train), y_train)

fake_test = fake.tail(bucket_test * 10)
real_test = real.tail(bucket_test * 10)

y_test = np.array([0] * bucket_test * 10 + [1] * bucket_test * 10)

classifier = newsClassifier()
classifier.parseDataframe(fake_train, "fake")
classifier.parseDataframe(real_train, "real")
classifier.smoothAll()
x_test = classifier.generateVector(pd.concat([fake_test, real_test]))


prediction = model.predict(np.asarray(x_test))
print(classification_report(y_test, prediction))


#  with https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset as dataset
#               precision    recall  f1-score   support
#
#            0       0.99      0.99      0.99      2000
#            1       0.99      0.99      0.99      2000
#
#     accuracy                           0.99      4000
#    macro avg       0.99      0.99      0.99      4000
# weighted avg       0.99      0.99      0.99      4000

# with https://www.kaggle.com/c/fake-news/data?select=train.csv as dataset

#               precision    recall  f1-score   support
#
#            0       0.72      0.86      0.79       700
#            1       0.83      0.67      0.74       700
#
#     accuracy                           0.77      1400
#    macro avg       0.78      0.77      0.76      1400
# weighted avg       0.78      0.77      0.76      1400

"""permutation importance to get feature importance"""
feature_names = ["title|fake", "title|real", "body|fake", "body|real", "body2|fake", "body2|real", "sent_title", "sent_body"]
permutation = PermutationImportance(model, random_state=0).fit(np.asarray(x_train), y_train)
output = eli5.explain_weights(permutation, feature_names=feature_names, top=8)
print(output)

# feature='title|fake', weight=0.02622857142857147,
# feature='title|real', weight=0.021014285714285763
# feature='body|fake', weight=0.01660000000000004,
# feature='body|real', weight=0.028071428571428636,
# feature='sent_body', weight=0.0033285714285714585,
# feature='sent_title', weight=0.0018285714285714682,
# feature='body2|fake', weight=0.0007142857142857561,
# feature='body2|real', weight=0.0005714285714286005
