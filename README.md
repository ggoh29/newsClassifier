# newsClassifier
A fake news classifier based on semantics analysis

Hexbridge Marshall Wace challenge 

Team twH1uxEQ, Bot that detects and reports disinformation on social media platforms

News Organiser based On Semantics Evaluation (NOOSE)

We have built a working model of a classifier that can predict fake news. It does this by looking at the semantics of the news article. 
Our dataset was taken from kaggle https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

We theorise that the semantics of a fake news article will be different from a real news articles, where a fake news articles will try to be more
persuasive and attention grabbing while a real news article is more objective or neutral.

On initialising the class Classifier, we generate 6 different transition matrices. 
- One for the title of real news articles (transition based on current state, x1 -> x2 - > x3 -> ...) (1)
- One for the title of fake news articles (transition based on current state, x1 -> x2 - > x3 -> ...) (2)
- One for the body of real news articles (transition based on current state, x1 -> x2 - > x3 -> ...) (3)
- One for the body of fake news articles (transition based on current state, x1 -> x2 - > x3 -> ...) (4)
- One for the body of real news articles (transition based on current state and previous state, x1^x2 -> x3, x2^x3 - > x4, ...) (5)
- One for the body of fake news articles (transition based on current state and previous state, x1^x2 -> x3, x2^x3 - > x4, ...) (6)

The idea is to calculate the probability that an article is fake or real given the list of parts of speech tokens the article contains.
The list of tokens is generated by using the nlp python nltk

So in order to find probability that an article is fake based on its title, P(fake | x1 -> x2 - > x3 -> ... ) we calculate the probability of our seen transition assuming the article is fake

Based on Bayes rule, P(fake | x1 -> x2 - > x3 -> ... ) = P(fake) * P(x1 -> x2 - > x3 -> ...| fake)/ P(x1 -> x2 - > x3 -> ...). However we want to compare the probability assuming the article is fake against the probability assuming it was real.

P(fake | x1 -> x2 - > x3 -> ... ) = P(fake) * P(x1 -> x2 - > x3 -> ...| fake)/ P(x1 -> x2 - > x3 -> ...) vs P(real| x1 -> x2 - > x3 -> ... ) = P(real) * P(x1 -> x2 - > x3 -> ...| real)/ P(x1 -> x2 - > x3 -> ...)

In order to simplify our calculations, we assume that P(fake) = P(real) and remove P(x1 -> x2 - > x3 -> ...) on both sides since that is just the average of the tranisition matrices.

So, we get P(x1 -> x2 - > x3 -> ...| fake) vs P(x1 -> x2 - > x3 -> ...| real)



Now comes the intresting part. 

At this point we have six values.

P(title = fake | x1 -> x2 - > x3 -> ...), P(title = real| x1 -> x2 - > x3 -> ...), (we scale these two such that the sum = 1)
P(body = fake | x1 -> x2 - > x3 -> ...), P(body = real| x1 -> x2 - > x3 -> ...), (we scale these two such that the sum = 1)
P(body = fake | x1^x2 -> x3, x2^x3 - > x4, ...), P(body = real| x1^x2 -> x3, x2^x3 - > x4, ...) (we scale these two such that the sum = 1).

We could take a majority vote based on the class that each pair gives a higher probability for, but we believe that each pair should be weighted. Therefore we add a gradient boosting classifier to our model.
We also add a 7th and 8th value which is the compound sentiment of the title and the body based on VADER semantics to add a bit more dimensionality. 

So for each article with a title and a body, we generate a vector of length 8 and feed that into our GBC.




When training the classifier, we have to be careful not to have information leakage, where we use a certain body and title to contribute to our transition matrix, and then calculate the vector of length 8 again for that article.
In order to avoid this, we used cross training which is demonstrated below

We break our training dateset into 10 equal buckets.
 
We use 9 buckets of articles to generate our transition matrices, and use these generated tranisition matric to get our vector of length 8 for the articles in the remaining bucket. Now we can generate a vector for each article while avoiding information leakage.

The gradient boosting classifier used is from sklearn and the hyperparameters are not fine tuned. 

Using a training set of 18000 real articles and 18000 fake articles and a test set of 2000 real articles and 2000 fake articles, we get the following results:

                  precision    recall  f1-score   support

               0       0.99      0.99      0.99      2000
               1       0.99      0.99      0.99      2000

        accuracy                           0.99      4000
       macro avg       0.99      0.99      0.99      4000
    weighted avg       0.99      0.99      0.99      4000

Which definetly looks suspicious. On closer inspection, the problem lies more with the data than our model. Other Kagglers have managed to achieve equally high prediction results using this dataset. 
We aim to change the dataset to get a better estimate of how well our model works.

Using a second dataset from https://www.kaggle.com/c/fake-news/data?select=train.csv, we get the following results:

                  precision    recall  f1-score   support

               0       0.73      0.83      0.77       700
               1       0.80      0.69      0.74       700

        accuracy                           0.76      1400
       macro avg       0.76      0.76      0.76      1400
    weighted avg       0.76      0.76      0.76      1400
 
Which looks more realistic and is a good sign. Out prediction accuracy is a lot better than random and our scores are pretty good. This means that the concept of this classifier can work.

Overall this is just a proof of concept of a machine learning model to classify fake news.  The important thing is that it must be able to classify an article based on the body and title since those are the most important parts.
In due time it will also be possible to extend this by incorporating it into a bot or website.
