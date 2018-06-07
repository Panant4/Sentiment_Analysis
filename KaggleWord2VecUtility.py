#!/usr/bin/env python

import re
import nltk

import pandas as pd
import numpy as np

from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer



class KaggleWord2VecUtility(object):

	
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def useless(clean_review):
	useless=["mitt romney","romney","obama","romneys","presidency","romneye","o b a m a","r o m n e y","mmitt","m i t t","teamobama","teambarack","barack","mitt","romneyi","romneyryan"]
    	review=str(clean_review)
    	for word in useless:
    	    review=review.replace(word,"entity")
	words = review.lower().split()
    	return words 	 
    @staticmethod
    def review_to_wordlist(review,remove_useless=False,stem=False,rem_hashtags=False,process=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
	tknzr = TweetTokenizer()

	new_temp=[]
	review=str(review)
	new_temp=re.sub('\\<.*?\\>', "", review)
    	new_temp=re.sub('\\http.*[\s]',"",new_temp)
   	new_temp=re.sub('\\http.*$',"",new_temp)
    	new_temp=re.sub('\\@(.*?)[\s]',"",new_temp)
    	new_temp=re.sub('\\@(.*?)$',"",new_temp)
    	#new_temp=re.sub('\\#(.*?)[\s]',"",new_temp)# if accuracy is less, try removing this
    	#new_temp=re.sub('\\#(.*?)$',"",new_temp)# if accuracy is less, try removing this
    	new_temp=re.sub(r'[^\w\s]',"",new_temp)

    	review_text=re.sub('\d+', "",new_temp)#does this hamper performance
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
	

        #3.2
	if  rem_hashtags:
		ar=[]
    		for w in words:
        		if w.startswith("#"):
            			ar.append(w)
		for val in ar:
			review_text=re.sub(val,'',review_text)
			
		words = review_text.lower().split()
		#words = tknzr.tokenize(review_text)
	a=[]
	def wordBreak( s, dict):
	        segmented = [True];
	        for i in range (0, len(s)):
	            segmented.append(False)
	            for j in range(i,-1,-1):
	                if segmented[j] and s[j:i+1] in dict:
	                    a.append(s[j:i+1])
	                    segmented[i+1] = True
	                    break
	        if segmented[len(s)]:
	            return a

	def str_join(*args):
    		return ' '.join(map(str, args))
	
	if process:	
    		ar=[]
		#with open('/home/pramod/posneg1.txt', 'r') as f:
    		#	myNames = [line.strip() for line in f]
    		for w in words:
        		if w.startswith("#"):
            			ar.append(w[1:])
				#print ar    		
		for val in ar:
        		ext=wordBreak(val,globl.vocab)
			if ext:
				ext1=' '.join(ext)
				review_text=review_text + ' ' + ext1

		words = review_text.lower().split()		

	if  stem:
		ar1=[]
		ps=PorterStemmer()
		lanc=LancasterStemmer()
		lemma = nltk.wordnet.WordNetLemmatizer()
		sno=SnowballStemmer("english", ignore_stopwords=True)
		words = [sno.stem(w) for w in words]
		words = [ps.stem(w) for w in words]


        # 4. Optionally remove stop words (false by default)
        if remove_useless:
		useless=["romney","obama","mitt"]
		for word in useless:
		        review_text=review_text.replace(word,"")

		words = review_text.lower().split()		
		
        #
        # 5. other regex

        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( KaggleWord2VecUtility.review_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
