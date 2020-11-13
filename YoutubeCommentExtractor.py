#!/usr/bin/env python
# coding: utf-8

# Reference:  https://python.gotrained.com/youtube-api-extracting-comments/

# In[25]:



import os
import pandas as pd
import pickle
import google.oauth2.credentials

import googleapiclient.discovery

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
# from github import Github


# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret"

# CLIENT_SECRETS_FILE = os.environ.get('client_secret')
# CLIENT_SECRETS_FILE: ${{secrets.CLIENT_SECRET}}
# github = Github(os.environ["GITHUB_TOKEN"])

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.

# SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
SCOPES=['https://www.googleapis.com/auth/youtube.readonly']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

def get_authenticated_service():
    credentials = None
    if os.path.exists('yumm.pickle'):
        with open('yumm.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()

        # Save the credentials for the next run
        with open('yumm.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials = credentials,cache_discovery=False)

def read_required_no_of_comments(url,requiredNoOfComments):
        """
        The function reads top level comments in multiples of 20. 
        """
        service = get_authenticated_service()
    # Get Video ID from URL entered
        splitURL=url.split("v=")
        videoId=splitURL[-1]

    #Get video title and no. of comments on the video
        requestVideo = service.videos().list(
            part="snippet,contentDetails,statistics",
            id=videoId
        )
        responseVideo = requestVideo.execute()
        totalComments=responseVideo['items'][0]['statistics']['commentCount']
        videoTitle=responseVideo['items'][0]['snippet']['title']
        print('Total comments: '+str(totalComments))

    # Initate empty list to store comment details
        comment = []
        comment_id = []
        author=[]
        reply_count = []
        like_count = []
        comment_date=[]

        if int(totalComments)>=1: 

    # Get comments for the required video
            requestComment = service.commentThreads().list(
            part="snippet",
            videoId=videoId)
            responseComment = requestComment.execute()

    # This will add first 20 comments into the list
            for item in responseComment['items']:
                        comment.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                        comment_id.append(item['snippet']['topLevelComment']['id'])
                        author.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
                        reply_count.append(item['snippet']['totalReplyCount'])
                        like_count.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
                        comment_date.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])

    # Check if another page exists. If yes, then read comments till there is no next page       
            while 'nextPageToken' in responseComment and len(comment_id)<int(requiredNoOfComments):
                    requestComment = service.commentThreads().list(
                    part="snippet",
                    videoId=videoId,
                    pageToken=responseComment['nextPageToken'])
                    responseComment = requestComment.execute()

    # Add items to list from next pages
                    for item in responseComment['items']:
                                comment.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                                comment_id.append(item['snippet']['topLevelComment']['id'])
                                author.append(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
                                reply_count.append(item['snippet']['totalReplyCount'])
                                like_count.append(item['snippet']['topLevelComment']['snippet']['likeCount'])
                                comment_date.append(item['snippet']['topLevelComment']['snippet']['publishedAt'])

    # Add all the values to a dataframe to return  
            doc=pd.DataFrame({ 
                          'Video Title':videoTitle,
                          'Comment': comment,
                          'CommentID': comment_id,
                          'Author':author,
                          'Replies': reply_count,
                          'Likes': like_count,
                          'CommentDate':comment_date})
    # Convert Datetime to Date
        doc['CommentDate']=pd.to_datetime(doc['CommentDate'])
        doc['CommentDate']=doc['CommentDate'].apply(lambda x: x.date())

        doc.drop_duplicates(subset=['Comment'],inplace=True)
        doc.reset_index(drop=True,inplace=True)
    # Save data 
        fileName="Comments_"+videoTitle
#         doc.to_excel("Data/"+fileName+".xlsx",index_label='CommentID',index=False)
        doc.to_csv("Score.csv",index_label='CommentID',index=False)
        return(doc)                    

# if __name__ == '__main__':
#     # When running locally, disable OAuthlib's HTTPs verification. When
#     # running in production *do not* leave this option enabled.
#     os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
#     service = get_authenticated_service()
    
#     url = input('Enter an URL: ')
#     requiredNoOfComments = input("Enter no. of comments you want to read: ")
        
#     read_required_no_of_comments(service,url,requiredNoOfComments)

