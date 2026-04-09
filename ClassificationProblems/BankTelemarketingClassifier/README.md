# Prediction of client subscription to a bank term deposit 

## Business Objective 

We have been given data from direct marketing campaigns (phone calls) of a Portuguese banking institution.
The goal of this project is to identify the most likely customers who would subscribe to a term deposit, so the bank can focus its marketing efforts and reduce wasted calls. Bank can make targetted calls, making sure interested customers are not missed and those not interested are not getting unnecessary calls.

In this usecase recall is more important than precision. Less Recall would mean,more false negatives, hence more missed subscriber. High recall is necessary in making sure interested customers are not missed.Low Precision would mean more false positives, hence more wrong predictions,that is uninterested customer is called, although this may add to redundant customer calls, this is lower business impact cost than missing interested customers.  (https://archive.ics.uci.edu/dataset/222/bank+marketing)

## Classification result 

After cleaning and data tranformation, various classification models were compared, namely Logistic Regression, SVM,Decision Trees and KNN.
It was determined that SVM gave the most balanced results.For this model, after applying the best params from GridSearch and applying a threshold of 0.77,  recall of 61% can be achieved and Precision of 41% can be achieved. This means that for every 100 customers that are actually interested, model is predicting 61 customers correctly. However for all the predicted customers, model is correct 41% of the time.
