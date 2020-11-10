#!/usr/bin/env python
#!python -m spacy download en_core_web_sm
# coding: utf-8

# In[ ]:


import nltk

nltk.download('wordnet')
nltk.download('punkt')

import spacy
spacy.load("en_core_web_sm")


