import pandas as pd
import html
import nltk
import re
from emoji import UNICODE_EMOJI

def text_preprocessing(df):
    df["Clean text"] = df["Text"].str.lower()
    df["Clean text"] = df["Clean text"].apply(remove_links)
    df["Clean text"] = df["Clean text"].apply(remove_html_entities)
    df["Clean text"] = df["Clean text"].apply(replace_username)
    return df


#Remove links, HTML special entities and username
def remove_links(text):
    #remove http & https
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
    #remove www
    text = re.sub('www.(?:[a-zA-Z]|[0-9]|[$#-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text, flags=re.MULTILINE)
    #remove emails
    text = re.sub('\S*@\S*\s?', '', text)
    return text

def remove_html_entities(text):
    text = html.unescape(text)
    return text

def replace_username(text):
    words = text.split(" ")
    result = []
    for word in words:
        if "@" in word:
            result.append("user")
        else:
            result.append(word)
    return " ".join(result)