# Sentment Analysis

A curated list of Sentiment Analysis methods, implementations and misc.

> Sentiment Analysis is the field of study that analyzes people's opinions, sentiments, evaluations, attitudes, and emotions from written languages. (Liu 2012)

The goal of this repository is to provide adequate links for scholars who want to research in this domain; And at the same time, be sufficiently accessible for developers who want to integrate sentiment analysis into their applications.

Sentiment Analysis (SA) happens at various levels: 
- Document-level SA evaluate sentiment of a single entity (i.e. a product) from a review document. 
- Sentence-level SA evaluate sentiment from a single sentence. 
- Aspect-level SA performs finer-grain analysis. For example, the sentence “the iPhone’s call quality is good, but its battery life is short.” evaluates two aspects: call quality and battery life, of iPhone (entity). The sentiment on iPhone’s call quality is positive, but the sentiment on its battery life is negative. (Liu 2012)

Most recent research focuses on the aspect-based approaches. But not all opensource implementations are caught up yet.

There are many different approaches to solve the problem. Some use Lexical methods, such as looking at the frequency of positive and negative words (from i.e. SentiWordNet) occuring in the given sentence. Some use Supervised Machine Learning, such as Naive Bayes and Support Vector Machine (SVM.) Some use Unsupervised Machine Learning, such as Latent Dirichlet Allocation (LDA) and word embeddings (Word2Vec). Recent work also apply Deep Learning methods such as Convolutional Neural Network (CNN) and Long Short-term Memory (LSTM), as well as their attention-based variants.

## Survey Papers 

Liu, Bing. "Sentiment analysis and opinion mining." Synthesis lectures on human language technologies 5.1 (2012): 1-167. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.244.9480&rep=rep1&type=pdf)

Vinodhini, G., and R. M. Chandrasekaran. "Sentiment analysis and opinion mining: a survey." International Journal 2.6 (2012): 282-292. [[pdf]](http://www.dmi.unict.it/~faro/tesi/sentiment_analysis/SA2.pdf)

Medhat, Walaa, Ahmed Hassan, and Hoda Korashy. "Sentiment analysis algorithms and applications: A survey." Ain Shams Engineering Journal 5.4 (2014): 1093-1113. [[pdf]](http://www.sciencedirect.com/science/article/pii/S2090447914000550)

## Baseline Systems

Wang, Sida, and Christopher D. Manning. "Baselines and bigrams: Simple, good sentiment and topic classification." Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Short Papers-Volume 2. Association for Computational Linguistics, 2012. [[pdf]](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)


## Open Source Implementations

### NodeJS
[thisandagain/sentiment]( https://github.com/thisandagain/sentiment): Lexical, Dictionary-based, AFINN-based.

[thinkroth/Sentimental](https://github.com/thinkroth/Sentimental) Lexical, Dictionary-based, AFINN-based.

### Java
[LingPipe](http://alias-i.com/): Lexical, Corpus-based, Supervised Machine Learning

[CoreNLP](https://github.com/stanfordnlp/CoreNLP): Supervised Machine Learning

[ASUM](http://uilab.kaist.ac.kr/research/WSDM11/): Unsupervised Machine Learning, Latent Dirichlet Allocation. [[paper]](http://www.cs.cmu.edu/~yohanj/research/papers/WSDM11.pdf)

### Python
[nltk](http://www.nltk.org/): [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analysis tool, Lexical, Dictionary-based, Rule-based. [[paper]](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)

[vivekn/sentiment](https://github.com/vivekn/sentiment): Supervised Machine Learning, Naive Bayes Classifier. [[paper]](https://arxiv.org/abs/1305.6143)

[xiaohan2012/twitter-sent-dnn](https://github.com/xiaohan2012/twitter-sent-dnn): Supervised Machine Learning, Deep Learning, Convolutional Neural Network. [[paper]](http://nal.co/papers/Kalchbrenner_DCNN_ACL14)

[kevincobain2000/sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier): Supervised Machine Learning, Naive Bayes Classifier, Max Entropy Classifier, SentiWordNet.

[pedrobalage/SemevalAspectBasedSentimentAnalysis](https://github.com/pedrobalage/SemevalAspectBasedSentimentAnalysis): Aspect-Based, Supervised Machine Learning, Conditional Random Field.

[ganeshjawahar/mem_absa](https://github.com/ganeshjawahar/mem_absa): Aspect-Based, Supervised Machine Learning, Deep Learning, Attention-based, External Memory. [[paper]](https://arxiv.org/abs/1605.08900)

### R
[timjurka/sentiment](https://github.com/timjurka/sentiment): Supervised Machine Learning, Naive Bayes Classifier.

### Golang
[cdipaolo/sentiment](https://github.com/cdipaolo/sentiment): Supervised Machine Learning, Naive Bayes Classifier. Based on [cdipaolo/goml](https://github.com/cdipaolo/goml).

### Ruby
[malavbhavsar/sentimentalizer](https://github.com/malavbhavsar/sentimentalizer): Lexical, Dictionary-based.

[7compass/sentimental](https://github.com/7compass/sentimental): Lexical, Dictionary-based.

## Contributing

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Steps to contribute:

- Make your awesome changes
- Submit pull request; if you add a new entry, please give a very brief explanation why you think it should be added.