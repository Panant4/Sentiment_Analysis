# Sentiment_Analysis
Analysis of twitter data of Barack Obama and Mitt Romney

Data: 7000 tweets of both Barack Obama and Mitt Romney \n
Attributes: Annotated Tweet, Class
Classes: 0:Neutral,1:positive,-1:negative,2:neither positive nor negative
Libraries:Pandas,Numpy,Scikit-Learn,Keras,NLTK

files: 1. KaggleWord2VecUtility.py- Preprocessing
2. Senti_romney.ipynb- Initial attempt to understand data
3.Obama_final.ipynb- Initial attempt to understand data
3. RNN_keras.ipynb- Initial attempt at Neural Networks
4. Romney_ana.ipynb- Model with highest accuracy

Summary:
->After removing useless classes, the resulting dataset consisted of unbalanced labels for both Obama and Romney tweets. This imbalance was handled using (*) given below.

Preprocessing
->First, preprocessing was done to remove HTML tags, hyperlinks and twitter handles
->Hashtags were replaced by the words themselves ("#obamarocks"->"obamarocks"). The words were not split further as valuable information provided by the hashtags could be lost
->NLTK's preprocessing APIs are used to optimize the data
->A combination of Porter and Snowball stemmer is used
->A dictionary of positive and negative words was created and run through the entire list of hashtags to extract sentiment information
->(*)A common name "Entity" was given instead of names that contained obama and romney. This helped get an overall better accuracy on the test data. The model learned the sentiment words better.

Machine Learning Pipeline

-> Tfidfvectorizer(ngrams:1-4) with hyperparameter tuning performed the best vectorization.
-> Multiple feature selection techniques like PCA and LDA were tried but not found to impact the accuracy
-> A tuned Linear SVM was found to give the best accuracy
-> Multiple RNN models were tried but were not found to perform better than the SVM model. Tuning of the parameters could have probably given better results.

