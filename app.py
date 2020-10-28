import warnings
warnings.filterwarnings("ignore")
                        
import pandas as pd
import streamlit as st
import dill 
import Helper
import YoutubeCommentExtractor
import importlib
importlib.reload(YoutubeCommentExtractor)
importlib.reload(Helper)
import time
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 200)
# import seaborn as sns
from matplotlib import pyplot as plt
import string as str
# import joblib
st.beta_set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align:center; position: relative; top: -20px;'>YouTube Comment Classifier</h1>", unsafe_allow_html=True)

#Expandable sidebar
exp=st.sidebar.beta_expander("About the project")
exp.write('Enter URL of a YouTube music video and no. of comments you want to analyze and hit Submit. The top level comments get scrapped from YouTube and the  classified into Spam and non-Spam (Ham) categories. ')

st.sidebar.markdown('[Github Repository](https://github.com/Preeti24/Youtube-comments)')

#Load the model
model = dill.load(open('fittedWinnerModel', "rb"))
# model = joblib.load(open('fittedWinnerModel', "rb"))

col1, col2,col3 = st.beta_columns([1.8,1,1])

#Get input
#https://www.youtube.com/watch?v=bsGp4A7u_LM
youTubeURL=col1.text_input(label='Enter YouTube music Video URL or use default',value='https://www.youtube.com/watch?v=EQfm-Qqy-wU')
noOfComments=col2.number_input(label='Enter no. of comments to read or use default',value=20)

#Function to scrap reviews for the given URL
@st.cache(suppress_st_warning=True)
def readReviews(youTubeURL,noOfComments):
    return YoutubeCommentExtractor.read_required_no_of_comments(youTubeURL,noOfComments)


if st.button(label='Submit'):
    try:
        with st.spinner('Running machine learning model...'):
           
            data=readReviews(youTubeURL,noOfComments);
            st.header(data['Video Title'][0])
            st.write("")
            data=pd.DataFrame(data={'Comment':data['Comment'],
                   'Classification':model.predict(data['Comment']),
                   'Prediction probability':model.predict_proba(data['Comment'])[:,1].round(3)})
            data['Classification']=data['Classification'].astype(int)
            
            #Display Count plot
            col1, col2 = st.beta_columns([2, 1])
            fig, ax = plt.subplots()
#             ax=sns.countplot(x='Classification',data=data,palette="Set3")
            plt.ylabel("No. of comments")
            plt.title("No. of Spam and Ham comments",fontdict={'fontsize':20})
            plt.xticks([0,1],labels=['Ham','Spam']);
            col1.pyplot(fig)
            
            #Display video thumbnail
            col2.video(youTubeURL)
            
            #Separate Ham data
            filt=data['Classification']==0
            dataHam=data[filt]
            dataHam['Prediction probability']=dataHam['Prediction probability'].apply(lambda x:1-x)
            dataHam.sort_values(by=['Prediction probability'],ascending=False,inplace=True)
            dataHam.reset_index(drop=True,inplace=True)
            dataHam.index = dataHam.index + 1

            #Separate Spam data
            dataSpam=data[~filt]
            dataSpam.sort_values(by=['Prediction probability'],ascending=False,inplace=True)
            dataSpam.reset_index(drop=True,inplace=True)
            dataSpam.index = dataSpam.index + 1

            #Display Spama dn Ham data in HTML markdown
            col3, col4 = st.beta_columns([1,1])
            col3.header('Spam Comments')
            dataSpamHtml=pd.DataFrame.to_html(dataSpam[['Comment','Prediction probability']],
                                  col_space=200,escape=False,justify='center',border=1,bold_rows=False,
                                  classes=['zebra','text-align: left']).replace('<tr>', '<tr align="left">')
            
            col3.markdown(dataSpamHtml,unsafe_allow_html=True)
            
            col4.header('Ham Comments')
            dataHamHtml=pd.DataFrame.to_html(dataHam[['Comment','Prediction probability']],
                                 col_space=200,escape=False,justify='center',border=1,bold_rows=False,
                                 classes=['text-align: left']).replace('<tr>', '<tr align="left">')
            
            col4.markdown(dataHamHtml,unsafe_allow_html=True)
        
    except:
        st.error("An error has occured")

