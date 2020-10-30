import warnings
warnings.filterwarnings("ignore")
                        
import pandas as pd
import numpy as np
import streamlit as st
import dill 
import Helper
import YoutubeCommentExtractor
import importlib
importlib.reload(YoutubeCommentExtractor)
importlib.reload(Helper)
import time
pd.set_option('display.max_columns', 100)
# pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_colwidth', -1)
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import string as str
import nltkModules
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import spacy spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
# import en_core_web_sm
# nlp = en_core_web_sm.load()

nlp = spacy.load('en_core_web_sm')
st.beta_set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align:center; position: relative; top: -20px;'>YouTube Comment Analyzer</h1>", unsafe_allow_html=True)

#Expandable sidebar
exp=st.sidebar.beta_expander("About the project")
exp.write('Enter URL of a YouTube music video and no. of comments you want to analyze and hit Submit. The top level comments get scrapped from YouTube and the  classified into Spam and non-Spam (Ham) categories. ')

st.sidebar.markdown('[Github Repository](https://github.com/Preeti24/Youtube-comments)')

#Load the model
model = dill.load(open('fittedWinnerModel', "rb"))
# model = joblib.load(open('fittedWinnerModel', "rb"))

col1, col2,col3 = st.beta_columns([1.8,1,1])

#Get input
#
# https://www.youtube.com/watch?v=EQfm-Qqy-wU
youTubeURL=col1.text_input(label='Enter YouTube music Video URL or use default',value='https://www.youtube.com/watch?v=QcIy9NiNbmo')
noOfComments=col2.number_input(label='Enter no. of comments to read or use default',value=20)

#Function to scrap reviews for the given URL
@st.cache(suppress_st_warning=True)
def readReviews(youTubeURL,noOfComments):
    return YoutubeCommentExtractor.read_required_no_of_comments(youTubeURL,noOfComments)

# Function for Sentiment Analysis
analyzer=SentimentIntensityAnalyzer()

@st.cache(suppress_st_warning=True)
def sentimentAnalysis(text):
    return analyzer.polarity_scores(text)['compound']


# Noun phrase Chuncking
def nounPhraseChunking(text):
    doc=nlp(text)
    l=[]
    for np in doc.noun_chunks:
        if (np.text).lower() not in stopwords.words('english') and len(np.text)>1\
        and ".com" not in np.text and "https" not in np.text and "http" not in np.text and\
        np.text not in ["br","<br",'️<br']:
            l.append(np.text)
    return l

#Return sorted dictionary of noun phrase chuncks
def nounDictionary(df):
    ncList=[]
    for x in df['NounChunks']:
        ncList.append(x)

    ncDictionary={}
    for nounChunk in ncList:
        for j in nounChunk:
            j=j.lower()
            if j not in ncDictionary:
                ncDictionary[j]=1
            else:
                ncDictionary[j]+=1

    return sorted(ncDictionary.items(), key=lambda x:x[1],reverse=True)

        
if st.button(label='Submit'):
#     try:
        with st.spinner('Running machine learning model...'):
           
            data=readReviews(youTubeURL,noOfComments);
            st.header(data['Video Title'][0])
            st.write("")
            #Spam and ham classification
            data=pd.DataFrame(data={'Comment':data['Comment'],
                   'Classification':model.predict(data['Comment']),
                   'Prediction probability':model.predict_proba(data['Comment'])[:,1].round(3)})
            data['Classification']=data['Classification'].astype(int)
            
          
            #Noun phrase chuncking
            data['NounChunks']=data['Comment'].apply(lambda x: nounPhraseChunking(x))
#             col1, col2 = st.beta_columns(2)
#             filt=data['Classification']==1
#             spamNC=pd.DataFrame(nounDictionary(data.loc[filt,['NounChunks']]),columns=['NounPhrases','Frequency'])
#             hamNC=pd.DataFrame(nounDictionary(data.loc[~filt,['NounChunks']]),columns=['NounPhrases','Frequency'])
#             col1.dataframe(spamNC['NounPhrases'].head(10))
#             col2.dataframe(hamNC['NounPhrases'].head(10))

            
            #Sentiment Analysis
            data['Polarity']=data['Comment'].apply(lambda x: sentimentAnalysis(x))
            data['Sentiment']=np.where(data['Polarity']==0,'UNC',np.where(data['Polarity']>0,'POS','NEG'))    
            
            df=data.groupby(['Classification','Sentiment']).size().reset_index().\
                            pivot(columns='Sentiment',index='Classification',values=0)
            df.reindex(['1','0'])
            df.fillna(0,inplace=True)
            df=df[['POS','UNC','NEG']]
            
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["limegreen","yellow","red"])
           
            #Display Count plot of Spam and Ham
            col5, col6 = st.beta_columns([2, 1])
            fig, ax = plt.subplots()
            df.plot(kind='bar', stacked=True,colormap=cmap,ax=ax)
            
            plt.ylabel("No. of comments")
            plt.title("No. of Spam and Ham comments",fontdict={'fontsize':20})
            plt.xticks([0,1],labels=['Ham','Spam'],rotation='horizontal');
            col5.pyplot(fig)
            
            #Display video thumbnail
            col6.video(youTubeURL)
            
            
            
            filt1=data['Sentiment']=='POS'
            filt2=data['Sentiment']=='NEG'
            spamNC=nounDictionary(data.loc[filt1,['NounChunks']])
            hamNC=nounDictionary(data.loc[filt2,['NounChunks']])
            
            col1, col2 = st.beta_columns(2)
            x=[]
            for item in hamNC:
                x.append(item[0])
            if len(x)>0:
                col1.subheader("Topics in Negative comments")
                col1.write(x[:10])
            
            y=[]
            for item in spamNC:
                y.append(item[0])
            if len(y)>0:
                col2.subheader("Topics in Positive comments")
                col2.write(y[:10])

            
            
            
            
            #Separate Ham data
            filt=data['Sentiment']=='POS'
            dataHam=data[filt]
            dataHam['Prediction probability']=dataHam['Prediction probability'].apply(lambda x:1-x)
            dataHam.sort_values(by=['Prediction probability'],ascending=False,inplace=True)
            dataHam.reset_index(drop=True,inplace=True)
            dataHam.index = dataHam.index + 1

            #Separate Spam data
            filt=data['Sentiment']=='NEG'
            dataSpam=data[filt]
            dataSpam.sort_values(by=['Prediction probability'],ascending=False,inplace=True)
            dataSpam.reset_index(drop=True,inplace=True)
            dataSpam.index = dataSpam.index + 1

            #Display Spama dn Ham data in HTML markdown
            col3, col4 = st.beta_columns([1,1])
            col3.header('Negative Comments')
            dataSpamHtml=pd.DataFrame.to_html(dataSpam[['Comment']],
                                  col_space=200,escape=False,justify='center',border=1,bold_rows=False,
                                  classes=['text-align: left']).replace('<tr>', '<tr align="left">')
            
            col3.markdown(dataSpamHtml,unsafe_allow_html=True)
#             col3.dataframe(dataSpam['Comment'].astype('object'))
            
            col4.header('Positive Comments')
            dataHamHtml=pd.DataFrame.to_html(dataHam[['Comment']],
                                 col_space=200,escape=False,justify='center',border=1,bold_rows=False,
                                 classes=['text-align: left']).replace('<tr>', '<tr align="left">')
            
            col4.markdown(dataHamHtml,unsafe_allow_html=True)
        
#     except:
#         st.error("An error has occured")

