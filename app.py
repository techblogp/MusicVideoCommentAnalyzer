import warnings
warnings.filterwarnings("ignore")
                        
import pandas as pd
import numpy as np
import re
import time
import string as str

import Helper
import YoutubeCommentExtractor
import nltkModules
import dill 
import importlib
importlib.reload(YoutubeCommentExtractor)
importlib.reload(Helper)
importlib.reload(nltkModules)



pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', -1)
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nltk.corpus import stopwords
from spacy.displacy.render import EntityRenderer
from IPython.core.display import display, HTML,Markdown
import spacy
from spacy import displacy
import streamlit as st
from streamlit import components


# import en_core_web_sm
# nlp=en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm')

st.beta_set_page_config(layout="wide")


# Title
st.markdown("<h1 style='text-align:center; position: relative; top: -30px; margin:0; padding: 0;'>YouTube Comment Analyzer</h1>", unsafe_allow_html=True)

inputURL, inputNumber,submitButton = st.beta_columns([1,.5,.5])

with st.beta_container():
#Get input
    youTubeURL=inputURL.text_input(label='Enter YouTube music Video URL or use default',value='https://www.youtube.com/watch?v=X8PwL3OdfGw')
    noOfComments=inputNumber.number_input(label='Enter no. of comments to read or use default',value=500)

#Expandable sidebar
exp=st.sidebar.beta_expander("About the App")
exp.write('This app provides text analytics on YouTube video comments for the given video URL. The top level comments get scrapped from YouTube and then classified by their sentiments and then into Spam and non-Spam (Ham) categories. \n\n This app also provides a list of top key phrases/topics in each of the Positive and Negative comments along with sample comments with those phrases. These phrase act as a good representation of comments\' content without having to read them one by one. ')

st.sidebar.markdown('[Github Repository](https://github.com/Preeti24/Youtube-comments)')

#Load the model
model = dill.load(open('fittedWinnerModel', "rb"))

#Function to scrap reviews for the given URL
@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def readReviews(youTubeURL,noOfComments):
    return YoutubeCommentExtractor.read_required_no_of_comments(youTubeURL,noOfComments)

# Function for Sentiment Analysis
analyzer=SentimentIntensityAnalyzer()

@st.cache(suppress_st_warning=True)
def sentimentAnalysis(text):
    return analyzer.polarity_scores(text)['compound']

#Noun phrase Chuncking- related functions
options = {
    'colors': { '': '#FF8800'}
}
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')  
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext
def nounPhraseChunking(text):
    doc=nlp(text)
    npList=[]
    for np in doc.noun_chunks:
        #remove english stopwords, phrases less that 2 characters long, urls by capturing .com and https and html tags
        if (np.text).lower() not in stopwords.words('english') and len(np.text)>1\
        and ".com" not in np.text and "https" not in np.text and\
        '<br' not in np.text and '&#' not in np.text and np.text!='br' and np.text!='&quot':
            
            npList.append((np.text, np.start_char, np.end_char, np.label_))
    return npList
def extractNPPhrases(x):    
    if x is None:
        return ''
    npPhraseList=[]
    for t in x:
        npPhraseList.append(t[0])
    return npPhraseList
def add_noun_phrases(df):
    """Create new column in data frame with noun phrases.
    
    Keyword arguments:
    df -- a dataframe object
    
    """
    df['NounChunks']=df['text'].apply(lambda x: nounPhraseChunking(x))
    return df
def custom_render(doc, df, column, topNP,options={}, page=False, minify=False, idx=0):
    
    renderer, converter = EntityRenderer, parseNounPhrases
    renderer = renderer(options=options)
    parsed = [converter(doc, df=df, idx=idx, column=column,topNP=topNP)]
    html = renderer.render(parsed, page=page, minify=minify).strip()
    st.markdown(get_html(html), unsafe_allow_html=True)
def parseNounPhrases(doc, df, idx, column,topNP):
    """Parse custom entity types that aren't in the original spaCy module.
    
    Keyword arguments:
    doc -- a spaCy nlp doc object
    df -- a pandas dataframe object
    idx -- index for specific query or doc in dataframe
    column -- the name of of a column of interest in the dataframe
    
    """
    if column in df.columns and df[column][idx] is not None:
        entities = df[column][idx]
        ents=[]
        for ent in entities:
            if ent[0].lower() not in topNP:
                continue
            ents.append({'start': ent[1], 'end': ent[2], 'label': ""})
            
    else:
        ents = []
    return {'text': doc.text, 'ents': ents, 'title': None}
def render_entities(idx, df,topNP, options={}, column='named_ents'):
    """A wrapper function to get text from a dataframe and render it visually in jupyter notebooks
    
    Keyword arguments:
    idx -- index for specific query or doc in dataframe (default 0)
    df -- a pandas dataframe object
    options -- various options to feed into the spaCy renderer, including colors
    column -- the name of of a column of interest in the dataframe (default 'named_ents')
    
    """
    text = df['text'][idx]
    custom_render(nlp(text), df=df, column=column, options=options, idx=idx,topNP=topNP)
def visualize_noun_phrases(text,topNP):
    """Create a temporary dataframe to extract and visualize noun phrases. 
    
    Keyword arguments:
    text -- the actual text source from which to extract entities
    
    """
    df = pd.DataFrame([text]) 
    df.columns = ['text']
    add_noun_phrases(df)
    column = 'NounChunks'
    render_entities(0, df,topNP=topNP, options=options, column=column)
def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 0rem; margin-bottom: 0.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)
def displayTopNComments(df,topNP):
    if df.shape[0]<5:
        x=df.shape[0]
    else:
        x=5
    for i in range(x):
        visualize_noun_phrases(df['Comment'][i],topNP)
def sampleComments(data):
    data['NounChunks']=data['Comment'].apply(lambda x: nounPhraseChunking(x))
    data['NounPhrase']=data['NounChunks'].apply(lambda x:extractNPPhrases(x))
    # Create a dictionary of noun phrases
    npDict={}
    nounPhrasePos={}
    nounPhraseNeg={}
    nounPhraseUnc={}
    
    for idx,nounPhrase in enumerate(data['NounPhrase']):
        if len(nounPhrase)==0:
            continue

        for np in nounPhrase:
            np=np.lower()
        
            if np not in npDict:
                npDict[np]=1
            else:
                npDict[np]+=1

            if data.iloc[idx].Sentiment=='POS':
                if np in nounPhrasePos:
                    nounPhrasePos[np]+=1
                else:
                    nounPhrasePos[np]=1

            if data.iloc[idx].Sentiment=='NEG':
                if np in nounPhraseNeg:
                    nounPhraseNeg[np]+=1
                else:
                    nounPhraseNeg[np]=1

            if data.iloc[idx].Sentiment=='UNC':
                if np in nounPhraseUnc:
                    nounPhraseUnc[np]+=1
                else:
                    nounPhraseUnc[np]=1
    df=pd.DataFrame(data=[npDict,nounPhraseNeg,nounPhrasePos,nounPhraseUnc]).transpose()
    df.rename(columns={0:'All',1:'Neg',2:'Pos',3:'Unc'},inplace=True)
    df.fillna(0,inplace=True)

    df['PosPercentage']=df.eval('Pos/All')
    df['NegPercentage']=df.eval('Neg/All')
    df['Diff']=df.eval('PosPercentage-NegPercentage')
    
    topPosNP=df[df['Diff']>0]
    topPosNP=topPosNP.sort_values(by=['Diff','All'],ascending=False)

    topNegNP=df[df['Diff']<0]
    topNegNP=topNegNP.sort_values(by=['Diff','All'],ascending=[True,False])
    
    if topNegNP.shape[0]>10:
        topNegNPList=topNegNP.index[:10].tolist()
    else:
        topNegNPList=topNegNP.index.tolist()

    if topPosNP.shape[0]>10:
        topPosNPList=topPosNP.index[:10].tolist()
    else:
        topPosNPList=topPosNP.index.tolist()    
        
        
    indexListPos=[]
    indexListNeg=[]
    for idx,nounPhrase in enumerate(data['NounPhrase']):
        if len(nounPhrase)==0:
            continue

        for np in nounPhrase:
            np=np.lower()

            if np in topPosNPList and data.iloc[idx].Sentiment=='POS':
                indexListPos.append(idx)
            if np in topNegNPList and data.iloc[idx].Sentiment=='NEG':
                indexListNeg.append(idx)
    samplePosComments=data.iloc[indexListPos].sort_values(by=['Polarity'],ascending=True)
    samplePosComments.reset_index(drop=True,inplace=True)
    
    sampleNegComments=data.iloc[indexListNeg].sort_values(by=['Polarity'],ascending=True)
    sampleNegComments.reset_index(drop=True,inplace=True)
    return samplePosComments,sampleNegComments,topPosNPList,topNegNPList

# This is to bring the button in center
submitButton.write("")                                            
submitButton.write("")   
if submitButton.button(label='Submit'):
    try:
        with st.spinner('Hold on!!!  Magic is hapenning...'):
           
            data=readReviews(youTubeURL,noOfComments);
            data['Comment']=data['Comment'].apply(lambda x: cleanhtml(x))

            #Spam and ham classification
            data=pd.DataFrame(data={'Comment':data['Comment'],
                    'CommentDate':data['CommentDate'],
                   'Classification':model.predict(data['Comment']),
                   'Prediction probability':model.predict_proba(data['Comment'])[:,1].round(3)})
            
#             data=pd.DataFrame(data={'Comment':data['Comment'],
#                                     'CommentDate':data['CommentDate'],
#                    'Classification':0,
#                    'Prediction probability':0})
            
            data['Classification']=data['Classification'].astype(int)
            
           
            #Sentiment Analysis
            data['Polarity']=data['Comment'].apply(lambda x: sentimentAnalysis(x))
            data['Sentiment']=data['Polarity'].apply(lambda x: 'POS' if x>0 else 'NEG' if x<0 else 'UNC')

            df=data.groupby(['Classification','Sentiment']).size().reset_index().\
                            pivot(columns='Sentiment',index='Classification',values=0)
            df.fillna(0,inplace=True)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["limegreen","yellow","red"])
           
            #Display Count plot of Spam and Ham
            col5, col6 = st.beta_columns([2,1])
            fig, (ax1, ax2) = plt.subplots(1, 2)
            df.plot(kind='bar', stacked=True,colormap=cmap,ax=ax1)
            plt.ylabel("No. of comments")
            plt.xticks([0,1],labels=['Ham','Spam'],rotation='horizontal');
            
            #Display comments over time
            df_overtime=data.groupby(['CommentDate','Sentiment']).size().reset_index().\
            pivot(columns='Sentiment',index='CommentDate',values=0)
            
            df_overtime.fillna(0,inplace=True)
            df_overtime.plot(kind='bar',stacked='True',colormap=cmap,ax=ax2);

            col5.pyplot(fig)

            #Display video thumbnail
            col6.video(youTubeURL)
            
            #----------------------------------------------------------------------
            #Noun phrase chuncking
            samplePosComments,sampleNegComments,topPosNPList,topNegNPList=sampleComments(data)
            samplePosComments=samplePosComments.drop_duplicates(subset=['Comment']).reset_index()
            sampleNegComments=sampleNegComments.drop_duplicates(subset=['Comment']).reset_index()
            
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown("<h2 style='text-align:left; position: relative; top: 0px; margin:0; padding: 0;'>Top POSITIVE things people are talking about</h2>", unsafe_allow_html=True)
                st.write(topPosNPList)
                st.markdown("<h3 style='text-align:left; position: relative; top: 0px; margin:0; padding: 0;'>Sample Comment</h2>", unsafe_allow_html=True)
                displayTopNComments(samplePosComments,topPosNPList)
            with col2:
                
                st.markdown("<h2 style='text-align:left; position: relative; top: 0px; margin:0; padding: 0;'>Top NEGATIVE things people are talking about</h2>", unsafe_allow_html=True)
                if len(topNegNPList)>0:
                    st.write(topNegNPList)
                    st.markdown("<h3 style='text-align:left; position: relative; top: 0px; margin:0; padding: 0;'>Sample Comments</h2>", unsafe_allow_html=True)
                else:
                    
                    st.markdown("<h3 style='text-align:center; position: relative; top: 0px; margin:0; padding: 0;'><br><br>People are very nice!<br> nobody has anything negative to say!!!</h3>", unsafe_allow_html=True)
                displayTopNComments(sampleNegComments,topNegNPList)
    except:
        st.error("An error has occured")

