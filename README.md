# 😀😄😂😭 Awesome Sentiment Analysis 😥😟😱😤  [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Curated list of Sentiment Analysis methods, implementations and misc.

> Sentiment Analysis is the field of study that analyzes people's opinions, sentiments, evaluations, attitudes, and emotions from written languages. (Liu 2012)

## Contents

<!-- TOC -->

- [Contents](#contents)
- [Objective](#objective)
- [Introduction](#introduction)
- [Survey Papers](#survey-papers)
- [Baseline Systems](#baseline-systems)
- [Resources and Corpora](#resources-and-corpora)
- [Open Source Implementations](#open-source-implementations)
    - [NodeJS](#nodejs)
    - [Java](#java)
    - [Python](#python)
    - [R](#r)
    - [Golang](#golang)
    - [Ruby](#ruby)
    - [CSharp](#csharp)
- [SaaS APIs](#saas-apis)
- [Web Apps](#web-apps)
- [Contributing](#contributing)

<!-- /TOC -->

## Objective

The goal of this repository is to provide adequate links for scholars who want to research in this domain; and at the same time, be sufficiently accessible for developers who want to integrate sentiment analysis into their applications.

## Introduction

Sentiment Analysis happens at various levels: 
- Document-level Sentiment Analysis evaluate sentiment of a single entity (i.e. a product) from a review document. 
- Sentence-level Sentiment Analysis evaluate sentiment from a single sentence. 
- Aspect-level Sentiment Analysis performs finer-grain analysis. For example, the sentence “the iPhone’s call quality is good, but its battery life is short.” evaluates two aspects: call quality and battery life, of iPhone (entity). The sentiment on iPhone’s call quality is positive, but the sentiment on its battery life is negative. (Liu 2012)

Most recent research focuses on the aspect-based approaches. But not all opensource implementations are caught up yet.

There are many different approaches to solve the problem. Lexical methods, for example, look at the frequency of words expressing positive and negative sentiment (from i.e. SentiWordNet) occurring in the given sentence. Supervised Machine Learning, such as Naive Bayes and Support Vector Machine (SVM), can be used with training data. Since training examples are difficult to obtain, Unsupervised Machine Learning, such as Latent Dirichlet Allocation (LDA) and word embeddings (Word2Vec) are also used on large unlabeled datasets. Recent works also apply Deep Learning methods such as Convolutional Neural Network (CNN) and Long Short-term Memory (LSTM), as well as their attention-based variants. You will find more details in the survey papers.

## Survey Papers 

Liu, Bing. "Sentiment analysis and opinion mining." Synthesis lectures on human language technologies 5.1 (2012): 1-167. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.244.9480&rep=rep1&type=pdf)

Vinodhini, G., and R. M. Chandrasekaran. "Sentiment analysis and opinion mining: a survey." International Journal 2.6 (2012): 282-292. [[pdf]](http://www.dmi.unict.it/~faro/tesi/sentiment_analysis/SA2.pdf)

Medhat, Walaa, Ahmed Hassan, and Hoda Korashy. "Sentiment analysis algorithms and applications: A survey." Ain Shams Engineering Journal 5.4 (2014): 1093-1113. [[pdf]](http://www.sciencedirect.com/science/article/pii/S2090447914000550)

## Baseline Systems

Wang, Sida, and Christopher D. Manning. "Baselines and bigrams: Simple, good sentiment and topic classification." Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Short Papers-Volume 2. Association for Computational Linguistics, 2012. [[pdf]](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)

Cambria, Erik, Daniel Olsher, and Dheeraj Rajagopal. "SenticNet 3: a common and common-sense knowledge base for cognition-driven sentiment analysis." Proceedings of the twenty-eighth AAAI conference on artificial intelligence. AAAI Press, 2014. [[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/download/8479/8602)

## Resources and Corpora

AFINN: List of English words rated for valence [[web]](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010)

SentiWordNet: Lexical resource devised for supporting sentiment analysis. [[web]](http://sentiwordnet.isti.cnr.it/) [[paper]](https://www.researchgate.net/profile/Fabrizio_Sebastiani/publication/220746537_SentiWordNet_30_An_Enhanced_Lexical_Resource_for_Sentiment_Analysis_and_Opinion_Mining/links/545fbcc40cf27487b450aa21.pdf)

GloVe: Algorithm for obtaining word vectors. Pretrained word vectors available for download [[web]](http://nlp.stanford.edu/projects/glove/) [[paper]](http://nlp.stanford.edu/pubs/glove.pdf)

SemEval14-Task4: Annotated aspects and sentiments of laptops and restaurants reviews. [[web]](http://alt.qcri.org/semeval2014/task4/) [[paper]](http://www.aclweb.org/anthology/S14-2004)

Stanford Sentiment Treebank: Sentiment dataset with fine-grained sentiment annotations [[web]](http://nlp.stanford.edu/sentiment/code.html) [[paper]](http://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)

Multidimensional Lexicon for Interpersonal Stancetaking [[web]](https://github.com/umashanthi-research/multidimensional-stance-lexicon) [[paper]](https://www.cc.gatech.edu/~jeisenst/papers/pavalanathan-acl-camera-ready.pdf)

## Open Source Implementations

The characteristics of each implementation are described.

_**Caveats**: A key problem in sentiment analysis is its sensitivity to the domain from which either training data is sourced, or on which a sentiment lexicon is built. [[♠]](http://www.springer.com/gp/book/9783319389707) Be careful assuming off-the-shelf implementations will work for your problem, make sure to look at the model assumptions and validate whether they’re accurate on your own domain [[♦]](https://lobste.rs/s/zsfqyk/curated_list_sentiment_analysis_methods/comments/ge671n#c_ge671n)._

### NodeJS
[thisandagain/sentiment]( https://github.com/thisandagain/sentiment): Lexical, Dictionary-based, AFINN-based.

[thinkroth/Sentimental](https://github.com/thinkroth/Sentimental) Lexical, Dictionary-based, AFINN-based.

### Java
[LingPipe](http://alias-i.com/): Lexical, Corpus-based, Supervised Machine Learning

[CoreNLP](https://github.com/stanfordnlp/CoreNLP): Supervised Machine Learning, Deep Learning

[ASUM](http://uilab.kaist.ac.kr/research/WSDM11/): Unsupervised Machine Learning, Latent Dirichlet Allocation. [[paper]](http://www.cs.cmu.edu/~yohanj/research/papers/WSDM11.pdf)

### Python
[nltk](http://www.nltk.org/): [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analysis tool, Lexical, Dictionary-based, Rule-based. [[paper]](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf)

[vivekn/sentiment](https://github.com/vivekn/sentiment): Supervised Machine Learning, Naive Bayes Classifier. [[paper]](https://arxiv.org/abs/1305.6143)

[xiaohan2012/twitter-sent-dnn](https://github.com/xiaohan2012/twitter-sent-dnn): Supervised Machine Learning, Deep Learning, Convolutional Neural Network. [[paper]](http://phd.nal.co/papers/Kalchbrenner_DCNN_ACL14)

[abdulfatir/twitter-sentiment-analysis](https://github.com/abdulfatir/twitter-sentiment-analysis): Sentiment analysis on tweets using Naive Bayes, SVM, CNN, LSTM, etc.

[kevincobain2000/sentiment_classifier](https://github.com/kevincobain2000/sentiment_classifier): Supervised Machine Learning, Naive Bayes Classifier, Max Entropy Classifier, SentiWordNet.

[pedrobalage/SemevalAspectBasedSentimentAnalysis](https://github.com/pedrobalage/SemevalAspectBasedSentimentAnalysis): Aspect-Based, Supervised Machine Learning, Conditional Random Field.

[ganeshjawahar/mem_absa](https://github.com/ganeshjawahar/mem_absa): Aspect-Based, Supervised Machine Learning, Deep Learning, Attention-based, External Memory. [[paper]](https://arxiv.org/abs/1605.08900)

[openai/generating-reviews-discovering-sentiment](https://github.com/openai/generating-reviews-discovering-sentiment): Deep Learning, byte mLSTM [[paper]](https://arxiv.org/abs/1704.01444)

[yiyang-gt/social-attention](https://github.com/yiyang-gt/social-attention): Deep Learning, Attention-based. Uses authors'
position in the social network to aide sentiment analysis. [[paper]](https://arxiv.org/pdf/1511.06052.pdf).

[thunlp/NSC](https://github.com/thunlp/NSC): Deep Learning, Attention-based. Uses user and production information.[[paper]](http://anthology.aclweb.org/D/D16/D16-1171.pdf).

### R
[timjurka/sentiment](https://github.com/timjurka/sentiment): Supervised Machine Learning, Naive Bayes Classifier.

### Golang
[cdipaolo/sentiment](https://github.com/cdipaolo/sentiment): Supervised Machine Learning, Naive Bayes Classifier. Based on [cdipaolo/goml](https://github.com/cdipaolo/goml).

### Ruby
[malavbhavsar/sentimentalizer](https://github.com/malavbhavsar/sentimentalizer): Lexical, Dictionary-based.

[7compass/sentimental](https://github.com/7compass/sentimental): Lexical, Dictionary-based.

### CSharp
[amrish7/Dragon](https://github.com/amrish7/Dragon): Supervised Machine Learning, Naive Bayes Classifier.


## SaaS APIs

* Google Cloud Natural Language API [[web]](https://cloud.google.com/natural-language/)
* IBM Watson Alchemy Language [[web]](https://www.ibm.com/watson/developercloud/alchemy-language.html)
* Microsoft Cognitive Service [[web]](https://www.microsoft.com/cognitive-services/en-us/text-analytics-api)
* Aylien [[web]](https://developer.aylien.com/text-api-demo)
* Indico [[web]](https://www.indico.io/)
* Rosette API [[web]](https://developer.rosette.com/)

## Web Apps

* Textalytic [[web]](https://www.textalytic.com)

## Contributing

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

Steps to contribute:

- Make your awesome changes
- Submit pull request; if you add a new entry, please give a very brief explanation why you think it should be added.
