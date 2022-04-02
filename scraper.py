import requests
from bs4 import BeautifulSoup
import json
import nltk
from nltk import *
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words("english")
stopwords.append('_')
nltk.download('wordnet')
tknzr = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

link = "https://www.thehindu.com"

articles = set()
article = requests.get(link)
article_content = article.content
soup_article = BeautifulSoup(article_content, 'html.parser')
for article in soup_article.find_all('a',href = re.compile("^https://www.thehindu.com\S*article[0-9]*\\.ece$")):
    articles.add(article.attrs['href'])

jsonArray = []
for a in articles:
    article = requests.get(a)
    soup_article = BeautifulSoup(article.content, 'html.parser')
    jsonarticle = {}

    authorsoup = soup_article.find(id=re.compile("content-body*"))
    if authorsoup is not None:
        body = authorsoup.find_all('p')
        bodytext = ""
        for p in body:
            bodytext += p.text + " "
        jsonarticle['body'] = bodytext

        title = soup_article.find('title')
        if title is not None:
            jsonarticle['title'] = title.text.strip()
        else:
            jsonarticle['title'] = "None"
        author = "None"
        date = "None"
        authorsoup = soup_article.find("div", class_="author-container")
        if authorsoup is not None:
            author = authorsoup.find('a', class_="auth-nm")
            if author is None:
                author = authorsoup.find('a', class_="person-nm")
                if author is not None:
                    author = author.text
                else:
                    author = "None"
            else:
                author = author.text
            date = authorsoup.find_all('span', class_="ksl-time-stamp")
            for d in date:
                if d is not None:
                    date = d.text.strip()
                else:
                    date = "None"
        else:
            authorsoup = soup_article.find("div", class_="authtitle")
            if authorsoup is not None:
                author = authorsoup.find('h', itemprop="name")
                if author is not None:
                    author = author.text.strip()
                else:
                    author = "None"
                date = authorsoup.find('h', class_="date",itemprop="datePublished")
                if date is not None:
                    date = date.text.strip()
                else:
                    date = "None"
        jsonarticle['author'] = author.strip()
        jsonarticle['date'] = date
        jsonArray.append(jsonarticle)

preprocessedList = []
for article in jsonArray:
    jsonarticle = article
    text = article['body']
    if len(text.strip()) > 0:
        finaltext = []
        finalWords = ""
        words = tknzr.tokenize(text)
        for word in words:
            if word not in stopwords and len(word) > 3:
                lem = lemmatizer.lemmatize(word)
                stem = ps.stem(lem)
                if stem not in stopwords and len(stem)> 3:
                    finaltext.append(stem)
        finalWords = ' '.join([word for word in finaltext])
        jsonarticle['preprocessed'] = finalWords.strip()
        preprocessedList.append(finaltext)

"""
Dumping the articles scraped into jso file
"""
with open("articles.json","w") as file:
    json.dump(jsonArray,file)
