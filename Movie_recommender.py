#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


movies = pd.read_csv('tmdb_5000_movies.csv')


# In[6]:


credits = pd.read_csv('tmdb_5000_credits.csv')


# In[7]:


movies.head()


# In[8]:


credits.head()


# In[9]:


movies = movies.merge(credits,on='title')


# In[10]:


movies.shape


# In[11]:


movies.isnull().sum()


# In[12]:


movies = movies[['id','title','genres','keywords','overview','cast','crew']]


# In[13]:


movies.head()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


import ast


# In[19]:


def convert(obj):
    gen = []
    for i in ast.literal_eval(obj):
        gen.append(i['name'])
    return gen


# In[20]:


movies['genres'] = movies['genres'].apply(convert)


# In[21]:


movies.head()


# In[22]:


movies['keywords']


# In[23]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[24]:


movies.head()


# In[25]:


movies['cast'][0]


# In[26]:


def convert_cast(obj):
    gen = []
    ct = 0
    for i in ast.literal_eval(obj):
        gen.append(i['name'])
        ct=ct+1
        if ct==3:
            break
    return gen


# In[27]:


movies['cast'] = movies['cast'].apply(convert_cast)


# In[28]:


movies['cast'][0]


# In[29]:


movies.head()


# In[30]:


movies['crew'][0]


# In[31]:


def convert_crew(obj):
    gen = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            gen.append(i['name'])
    
    return gen


# In[32]:


movies['crew'].apply(convert_crew)


# In[33]:


movies['crew'] = movies['crew'].apply(convert_crew)


# In[34]:


movies.head()


# In[35]:


def convert_overview(obj):
    gen = obj.split()
    
    return gen


# In[36]:


movies['overview'] = movies['overview'].apply(convert_overview)


# In[37]:


movies.head()


# In[38]:


def nospace(obj):
    for i in range(0,len(obj)):
        obj[i] = obj[i].replace(" ","")
    return obj


# In[39]:


movies["genres"] = movies["genres"].apply(nospace)
movies["keywords"] = movies["keywords"].apply(nospace)
movies["cast"] = movies["cast"].apply(nospace)
movies["crew"] = movies["crew"].apply(nospace)


# In[40]:


movies.head()


# In[41]:


movies['tags'] = movies['overview']+movies["genres"]+movies["keywords"]+movies["cast"]+movies["crew"]


# In[42]:


movies.head()


# In[43]:


new_movies = movies[['id','title','tags']]


# In[44]:


new_movies.head()


# In[45]:


def tagstr(obj):
    s=""
    for word in obj:
        s = s + word + " "
    return s


# In[46]:


new_movies["tags"] = new_movies["tags"].apply(tagstr)


# In[47]:


new_movies.head()


# In[48]:


new_movies['tags'][0]


# In[49]:


new_movies['tags'] = new_movies['tags'].apply(lambda x:x.lower())


# In[50]:


new_movies.head()


# In[51]:


import nltk


# In[52]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[53]:


def stemfunc(text):
    s=""
    for i in text.split():
        s = s + ps.stem(i) + " "
    return s


# In[54]:


new_movies['tags'] = new_movies['tags'].apply(stemfunc)


# In[55]:


new_movies['tags'][0]


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[57]:


words = cv.fit_transform(new_movies['tags']).toarray()


# In[58]:


words[0]


# In[59]:


cv.get_feature_names()


# In[60]:


from sklearn.metrics.pairwise import cosine_similarity


# In[61]:


similarity = cosine_similarity(words)


# In[62]:


similarity[0]


# In[63]:


def recommend(movie):
    
    ind = new_movies[new_movies['title']==movie].index[0]
    rec_movies = sorted(list(enumerate(similarity[ind])),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in rec_movies:
        print(new_movies['title'][i[0]])


# In[64]:


recommend('The Dark Knight')


# In[65]:


new_movies[new_movies['title']=='Aliens vs Predator: Requiem'].index[0]


# In[66]:


import pickle


# In[67]:


pickle.dump(new_movies,open('movies.pkl','wb'))


# In[68]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




