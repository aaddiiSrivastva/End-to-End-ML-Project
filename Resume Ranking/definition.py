import numpy as np
import pandas as pd
import string

import nltk
# nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

from sklearn.feature_extraction.text import CountVectorizer
CountVector=CountVectorizer()


import warnings
warnings.filterwarnings("ignore")



def reqDocExtraction(pdf_path):
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            yield text
            # close open handles
            converter.close()
            fake_file_handle.close()

alltext = []
def extract_text(pdf_path):
    for page in reqDocExtraction(pdf_path):
        alltext = []
        alltext.append(page)
        return alltext

def cleaningData(dataToClean):
    reqWords=[]
    lemmatizer = WordNetLemmatizer()
    for line in dataToClean:
        words = nltk.word_tokenize(line)
        lowWords = [w.lower() for w in words]
        lemWords = [lemmatizer.lemmatize(word) for word in lowWords if word not in set(stopwords.words('english'))]
        table=str.maketrans('','',string.punctuation)
        strpp=[w.translate(table) for w in lemWords]
        words=[word for word in strpp if word.isalpha()]
        joinedTokens = ' '.join(words)
        reqWords.append(joinedTokens)
    return reqWords

def creatingVector(data):
    # CountVector=CountVectorizer()
    bagOfWords=CountVector.fit_transform(data)
    countVectDF = pd.DataFrame(bagOfWords.toarray(),columns=CountVector.get_feature_names())
    # return countVectDF.to_dict('split')
    return countVectDF

def converToVec(data):
    # CountVector1=CountVectorizer()
    test_vector = CountVector.transform(data)
    resumeVector = pd.DataFrame(test_vector.toarray(),columns=CountVector.get_feature_names()).head(10)
    # return resumeVector.to_dict('split')
    return resumeVector

