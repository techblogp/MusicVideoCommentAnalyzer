import warnings
warnings.filterwarnings("ignore")
import nltkModules
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(15,5)
pd.set_option('display.max_columns', 100)

import nltk
from scipy import stats
import re
from  scipy.stats import ttest_ind
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve,fbeta_score,make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from cf_matrix import make_confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score,train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

import emoji
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import unicodedata
from nltk.stem import WordNetLemmatizer
import textstat


class Thresholder:
    """This function computes Precision, Recall, Accuracy,
        F1 Score and Fbeta Score for various values for threshold 
        between 0 and 1.
        
        
        Parameters
        ----------
        gs : estimator
        X : Predictor Variables
        y : Target Variable
        beta : value of beta to use for calculating Fbeta Score
        plotChart: Boolean (Default=False)
        
        Returns
        -------
        Dataframe with computed values for all the mectric for
        100 values of threshold. 
    """
    
    def __init__(self,gs,X,y,plotOptimalThreshold=False,plotROCCurve=False):  
        self.gs=gs
       
        self.y=y
        self.plotOptimalThreshold=plotOptimalThreshold
        self.plotROCCurve=plotROCCurve
        self.X=X
        
    def fit(self,beta=1):
        
        self.beta=beta
        self.metric_df=self.findOptimalThreshold(self.gs,self.X,self.y,self.beta)
        
        if self.plotOptimalThreshold==True:
            self.plotOptimalThreshold_(self.metric_df)
        
        if self.plotROCCurve==True:
            self.plotROCCurve_(self.gs,self.X,self.y)
            
        return self
    
    def findOptimalThreshold(self,gs,X,y,beta):
        
        predict_proba=self.gs.predict_proba(X)[:,1]
        
        thresholdRange=list(set(predict_proba))
#         thresholdRange=np.arange(0.1,1,0.1)
            
        precision_list=[]
        f1Score_list=[]
        fbetaScore_list=[]
        accuracy_list=[]
        recall_list=[]  
        threshold_list=[]
        for t in thresholdRange:
            predict_newThreshold=[]
            for prob in predict_proba:
                if prob<t:
                    predict_newThreshold.append(0)
                else:   
                    predict_newThreshold.append(1)
            cm=confusion_matrix(y.astype(int),predict_newThreshold)
            tp = cm[0][0] 
            fp = cm[0][1] 
            fn = cm[1][0] 
            tn = cm[1][1]
            precision=tp/(tp+fp)
            recall=tp/(tp+fn)
            F1=2*(precision*recall)/(precision+recall)
            FBeta=((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
            accuracy=(tp + tn) /(tp + tn + fp + fn)
            
            precision_list.append(precision)
            recall_list.append(recall)
            f1Score_list.append(F1)
            fbetaScore_list.append(FBeta)
            accuracy_list.append(accuracy)   
            threshold_list.append(t)
            
        metric_df=pd.DataFrame(data={'Threshold':threshold_list,
                         'Precision':precision_list,
                         'Recall':recall_list,
                         'F1':f1Score_list,
                         'FBeta':fbetaScore_list,
                         'Accuracy':accuracy_list})
        metric_df['Beta']=self.beta
                      
        metric_df.reset_index(drop=True,inplace=True)
#         metric_df.fillna(0,inplace=True)
        return metric_df
        
    def plotOptimalThreshold_(self,metric_df):
            xmark=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['Threshold'].iloc[0]
            ymark=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['FBeta'].iloc[0]
            beta=metric_df[metric_df['FBeta']==max(metric_df['FBeta'])]['Beta'].iloc[0]

            text= "FBeta={:.4f}".format(ymark)
            sns.lineplot(x='Threshold',y='FBeta',data=metric_df,label='Fbeta Score')
            sns.lineplot(x='Threshold',y='Precision',data=metric_df,label='Precision')
            sns.lineplot(x='Threshold',y='Recall',data=metric_df,label='Recall')
            plt.axvline(x=xmark,ymin=0,ymax=1,color='red')
            plt.annotate(s=text,xy=(xmark,ymark))
            plt.ylabel('Metrics')
            title='Threshold vs Metrics'
            plt.title(title);
            
    def plotROCCurve_(self,gs,X,y):
            fpr, tpr, thresholds=roc_curve(y,gs.predict_proba(X)[:,1])
            self.false_positive_rate=fpr
            self.true_positive_rate=tpr
            self.thresholds=thresholds

            plt.plot(fpr, tpr)
            title='ROC curve for '+ str(gs.best_estimator_.named_steps['clf']).split('(')[0]
            plt.title(title)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.show()
        

class TransformInput(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Learn how to transform data based on input data, X.
        """
        return self

    def transform(self, X):
        """
        Transform X into a new dataset, Xprime and return it.
        """
        X=pd.DataFrame(X)
        def countCaps(comment):
            count=0
            for c in comment:
                if c.isupper():
                    count+=1
            return round(count*100/len(comment),2)
        X['%OfUpperCaseLetters']=X['Comment'].apply(countCaps)
        
        pattern='https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}'
        X['NoOfURL']=X['Comment'].apply(lambda x: len(re.findall(pattern,x)))

        X['NoOfWords']=X['Comment'].apply(lambda x: (len(word_tokenize(x))))
        
        X['AvgSentenceLength']=X['Comment'].apply(lambda x: textstat.avg_sentence_length(x))
        
        X['TextStandard']=X['Comment'].apply(lambda x: textstat.text_standard(x,float_output=True))
        
        self.X=X
        return self.X

class GridSearcher:
    """
    This function performs grid search using given text 
    vectorizer and estimator for predictor and target variables.
    
    
    Parameters
    ----------
    X: Predictor Variables
    y: target Variable
    n_splits: Number of folds. Must be at least 2.
    vectorizer: text vectorizer
    clf: estimator
    param_grid: The parameter grid to explore, as a dictionary 
    mapping estimator parameters to sequences of allowed values.
        
    Returns
    -------
    None
    """
    def __init__(self,n_splits,vectorizer,clf,param_grid):
        self.n_splits=n_splits
        self.vectorizer=vectorizer
        self.clf=clf
        self.param_grid=param_grid
   
    def fit(self,X,y):
        return self.findBestEstimator(X,y,self.n_splits,self.vectorizer,self.clf,self.param_grid)

    

    def findBestEstimator(self,X,y,n_splits,vectorizer,clf,param_grid):   
        
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        
        get_text_data = FunctionTransformer(lambda x: x['Comment'], validate=False)
        get_numeric_data = FunctionTransformer(lambda x: x[['%OfUpperCaseLetters', 
                                                            'NoOfURL', 
                                                            'NoOfWords', 
                                                            'AvgSentenceLength',
                                                            'TextStandard']], validate=False)
#        Pipepline with FeatureUnion. Feature Union joins the text features and meta features generated during Feature engineering
        pipe = Pipeline([('transformInput',TransformInput()),
                        ('features', FeatureUnion([
                                                   ('meta_features', Pipeline([
                                                                               ('selector', get_numeric_data)
                                                                               ])),
                                                   ('text_features', Pipeline([
                                                                               ('selector', get_text_data),
                                                                               ('vectorizer', vectorizer)
                                                                               ]))
                                                  ])),
                        ('clf', clf)
        ])
        
        gs = RandomizedSearchCV(pipe,param_grid,n_jobs=-1, cv=cv, verbose=0,return_train_score=True,scoring='roc_auc')
        gs.fit(X,y);
        
        self.pipe=pipe
        self.cv=cv
        self.best_params=gs.best_params_
        self.best_score=gs.best_score_
        self.fittedWinnerModel=gs
        self.X=X
        
        return self

def lemmatize(cleanTextList):       
        wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
        pos_tagged_text = nltk.pos_tag(cleanTextList)
        lemmatizer=WordNetLemmatizer()

        return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def TextPreprocessor(comment): 
    """
     This function takes list of text and performs a 
        list of processing steps and return text ready 
        to be fed to machine learning model. 
        
        The list of cleaning steps:
        1. Remove URLs
        2. Convert emojis to text
        3. Convert numbers to '9's
        4. Remove non-english text
        5. Remove english stop words
        6. Remove punctuations
        7. Lemmatize text
        
        
        Parameters
        ----------
        CleanWordList : List
        
        Returns
        -------
        List of words
    """
    pattern1='(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    comment=re.sub(pattern1,' ',comment)
    comment=re.sub('<[^<]+?>', '', comment)
    comment=comment.replace(u'\ufeff', '')
    comment=emoji.demojize(comment, delimiters=("", ""))
    comment=comment.lower()
    comment=re.sub(r'\d','9',comment)
    comment = re.sub("([^\x00-\x7F])+"," ",comment)
    wordList=word_tokenize(comment)
    wordList=[word for word in wordList if word not in stopwords.words('english')]
    wordList=[word for word in wordList if word not in string.punctuation]
    cleanedTextList=[unicodedata.normalize('NFKD',word).encode('ascii', 'ignore').decode('utf-8', 'ignore') \
                             for word in wordList]
    lemmatizedText=lemmatize(cleanedTextList)
    return lemmatizedText
 
