from file_reader import FileReader
from collections import Counter
import random
import nltk
import numpy as np
import codecs
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import csv
import collections
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import plotly.plotly as py
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.learning_curve import learning_curve
from sklearn.feature_selection import VarianceThreshold

#%matplotlib inline

class FakeReviewsDetection(object):
    def __init__(self, genuine_file, fake_file):
        # Get True Reviews
            
        self.true_filereader = FileReader(genuine_file, "True") # new object
        self.true_filereader.parse_file()
        self.genuine_reviews = self.true_filereader.get_review_list()
        # print self.genuine_reviews

        # Get Fake Reviews
        self.fake_file_reader = FileReader(fake_file, "Fake") # new object
        self.fake_file_reader.parse_file()
        self.fake_reviews = self.fake_file_reader.get_review_list()
        #print self.fake_reviews[0]
        # Merge both the Reviews
        self.combined_reviews = []
        self.combined_reviews.extend(self.genuine_reviews)
        self.combined_reviews.extend(self.fake_reviews)
        
        '''all_reviews_docs = []
        for f_review in self.combined_reviews:
            all_reviews_docs.append(f_review["review"])
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_reviews_docs)'''
        
        self.Train_Classifier()

    def Train_Classifier(self):
        features = []
        correct_labels = []
        
        # getting features
        
        TruthfullAll = pd.read_csv('TruthfulALL.csv')
        Truthful_labels = (np.full((1,800),1)).tolist()
     
        FakeAll = pd.read_csv('FakeALL.csv')
        Fake_labels = (np.full((1,800),0)).tolist()
    
        data = TruthfullAll.copy()
        data = data.append(FakeAll,ignore_index = True)
        data = data.astype(float)
        dataMatrix = data.as_matrix()
        
        labels = Truthful_labels[0]
        labels.extend(Fake_labels[0])
        
        corpus = self.get_corpus(self.combined_reviews)
        
        length = 0
        for line in corpus:
            token = line.split()
            length +=len(token)
        
        print "number of terms in corpus : ",length
        
        print "creating n-grams....."
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 0, stop_words = 'english')
        
        tfidf_matrix =  tf.fit_transform(corpus)
        print "the size unigram 1:", tfidf_matrix.shape
        #Feature reduction using SVD
        svd = TruncatedSVD(n_components=100, random_state=50)
        U = svd.fit_transform(tfidf_matrix)
        print "the size unigram 2:", U.shape
         
        print "creating features......"
        count = 0
        
        for review_object in self.combined_reviews:
            # Get review tex
            #print "feat # ",count
            feat = []
            feat.extend(dataMatrix[count,:])
            feat.extend(U[count,:])
            count+=1
            
            
            features.append(feat)
        
        
        X = np.matrix(features)
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minmax = min_max_scaler.fit_transform(X)
        
        y = np.asarray(labels)
        
        # Selecting important features
        print "Original Features : ",X_minmax.shape
        
        columns = data.columns
        
        self.importantFeatures(X_minmax,y,columns)
        
        # Plotting word cloud from the corpus
        
        for index in range(len(self.genuine_reviews)):
            self.genuine_reviews[index]["review"] = self.genuine_reviews[index]["review"].replace("hotel","")
            self.genuine_reviews[index]["review"] = self.genuine_reviews[index]["review"].replace("Chicago","")
        
        for index in range(len(self.fake_reviews)):
            self.fake_reviews[index]["review"] = self.fake_reviews[index]["review"].replace("hotel","")
            self.fake_reviews[index]["review"] = self.fake_reviews[index]["review"].replace("Chicago","")
        
        print "\n\n Word Cloud of Genuine reviews based on the frequencies of words"
        self.wordCloud(self.genuine_reviews)
        print "\n\n Word Cloud of Fake reviews based on the frequencies of words"
        self.wordCloud(self.fake_reviews)
        
        X_Train, X_Test, y_Train, y_Test = train_test_split(X_minmax, y, test_size=0.2, random_state=0)
        
        skf = cross_validation.StratifiedKFold(y_Train, 4)
        nb_accuracy = []
        lg_accuracy = []
        svm_accuracy = []
        rd_forest = []
        count = 0
        
        for train,test in skf:
            count = count+1
            print "\n\n\nIteration # ",count
            
            X_train, X_test, y_train, y_test = X_Train[train], X_Train[test], y_Train[train], y_Train[test]
            
            
            print "\nRandom Forest Classifier..."
            forest = RandomForestClassifier(n_estimators = 50)
            forest = forest.fit(X_train, y_train)
            title = "Learning Curves (Random Forest), Iteration ",count
            # Cross validation with 100 iterations to get smoother mean test and train
            # score curves, each time with 20% data randomly selected as a validation set.
            cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,
                                               test_size=0.2, random_state=0)
            self.plot_learning_curve(forest, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            plt.show()

            y_pred = forest.predict(X_test)

            rd_forest.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            
            print "\nTraining Naive Bayes..."
            clf = MultinomialNB().fit(X_train, y_train)
            title = "Learning Curves (Naive Bayes), Iteration ",count
            # Cross validation with 100 iterations to get smoother mean test and train
            # score curves, each time with 20% data randomly selected as a validation set.
            cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,
                                               test_size=0.2, random_state=0)
            self.plot_learning_curve(clf, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            plt.show()
            y_pred = clf.predict(X_test)
            
            nb_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            print "\nTraining Logistic Regression..."
            log_reg_classifier = LogisticRegression().fit(X_train, y_train)
            title = "Learning Curves (Logistic Regression), Iteration ",count
            # Cross validation with 100 iterations to get smoother mean test and train
            # score curves, each time with 20% data randomly selected as a validation set.
            cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,
                                               test_size=0.2, random_state=0)
            self.plot_learning_curve(log_reg_classifier, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

            plt.show()
            y_pred = log_reg_classifier.predict(X_test)
            
            lg_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
            print "\nTrainig SVM..."
            svm_classifier = SVC().fit(X_train, y_train)
            title = "Learning Curves (SVM), Iteration ",count
            # Cross validation with 100 iterations to get smoother mean test and train
            # score curves, each time with 20% data randomly selected as a validation set.
            cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=100,
                                               test_size=0.2, random_state=0)
            self.plot_learning_curve(svm_classifier, title, X_train, y_train, ylim=(0.1, 2.01), cv=cv, n_jobs=4)

            plt.show()
            y_pred = svm_classifier.predict(X_test)
            
            svm_accuracy.append(metrics.accuracy_score(y_test, y_pred))
            print("accuracy:", metrics.accuracy_score(y_test, y_pred))
            
        print "\n\n#####Random Forest#####"
        print "Average Training Accuracy : ",np.array(rd_forest).mean()
        y_pred = forest.predict(X_Test)
        rf_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",rf_a
        #print "Precision Accuracy : ",metrics.precision_score(y_Test, y_pred)
        #print "Recall_score : ",metrics.recall_score(y_Test, y_pred)
        #print "f1_score : ",metrics.f1_score(y_Test, y_pred)
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        
        print "\n\n#####Naive Bayes#####"
        print "Average Training Accuracy : ",np.array(nb_accuracy).mean()
        y_pred = clf.predict(X_Test)
        nb_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",nb_a
        #print "Precision Accuracy : ",metrics.precision_score(y_Test, y_pred)
        #print "Recall_score : ",metrics.recall_score(y_Test, y_pred)
        #print "f1_score : ",metrics.f1_score(y_Test, y_pred)
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        
        print "\n\n#####Logistic Regression#####"
        print "Average Training Accuracy : ",np.array(lg_accuracy).mean()
        y_pred = log_reg_classifier.predict(X_Test)
        lr_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",lr_a
        #print "Precision Accuracy : ",metrics.precision_score(y_Test, y_pred)
        #print "Recall_score : ",metrics.recall_score(y_Test, y_pred)
        #print "f1_score : ",metrics.f1_score(y_Test, y_pred)
        print(metrics.classification_report(y_Test, y_pred,
                                    target_names=['Genuine', 'Fake']))
        
        print "\n\n#####Support Vector Machine#####"
        print "Average Training Accuracy : ",np.array(svm_accuracy).mean()
        y_pred = svm_classifier.predict(X_Test)
        svm_a = metrics.accuracy_score(y_Test, y_pred)
        print "Testing Accuracy : ",svm_a
        #print "Precision Accuracy : ",metrics.precision_score(y_Test, y_pred)
        #print "Recall_score : ",metrics.recall_score(y_Test, y_pred)
        #print "f1_score : ",metrics.f1_score(y_Test, y_pred)
        print(metrics.classification_report(y_Test, y_pred,
                                        target_names=['Genuine', 'Fake']))
        
        
        accuracy = []
        accuracy.append(nb_a)
        accuracy.append(svm_a)
        accuracy.append(lr_a)
        accuracy.append(rf_a)
        self.plotBarChart(accuracy)
        

    
    def get_corpus(self, reviews):
        words = []
        corpus = []
        for review in reviews:
            words.append(review["review"])
        
        return words

    
    def importantFeatures(self,X,Y,names):
        from sklearn.feature_selection import RFE, f_regression
        import operator
            
        f, pval  = f_regression(X, Y, center=True)
        ranks = rank_to_dict(f, names)
        
        print "printing top features according to their importance (rank)...."
        sorted_ranks = sorted(ranks.items(), key=operator.itemgetter(1))
        
        print sorted_ranks
    
        
    
    def wordCloud(self,data):
        corpus = ""
        for review in data:
            corpus+=review["review"]
        wordcloud = WordCloud(background_color='white',max_font_size=40, relative_scaling=.5,width=600,
                          height=400,stopwords=None).generate(corpus)
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()
    
    def plot_learning_curve(self,estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
   
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt
    
    def plotBarChart(self,accuracy):
        objects = ('Naive Bayes', 'SVM', 'Logistic Regression', 'Random Forest')
        y_pos = np.arange(len(objects))
        performance = accuracy
         
        plt.bar(y_pos, performance, align='center', alpha=0.3)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy')
         
        plt.show()

        
        
def rank_to_dict(ranks, names, order=1):
    minmax = preprocessing.MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))    
    
    
    # small dataset
#a = FakeReviewsDetection("op_spam_v1.4_merged/Genuine.txt","op_spam_v1.4_merged/Fake.txt")

       # All dataset 798 fake , 800 Genuine
#a = FakeReviewsDetection("op_spam_v1.4_merged/TruthfulALL.txt","op_spam_v1.4_merged/FakeALL.txt")
        #All datset 800:800
a = FakeReviewsDetection("TruthfulNew.txt","FakeNew.txt")
 