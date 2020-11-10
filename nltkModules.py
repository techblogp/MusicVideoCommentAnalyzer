#!/usr/bin/env python
#!python -m spacy download en_core_web_sm
# coding: utf-8

# In[ ]:


import nltk

nltk.download('wordnet')
nltk.download('punkt')

import spacy
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
# spacy.load('en_core_web_sm')

import pip
failed = pip.main(["install","en_core_web_sm"])

