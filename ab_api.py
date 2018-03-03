
## Building the NLP Engine for Albert Pinto

import pickle
import os
import sys
import Tweet_tokenizer as tt
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
def lemmatization_for_CV(message):
    message = unicode(message, 'utf-8').lower()
    words =  tt.tokenize(message)
    return [wnl.lemmatize(word.encode('ascii','ignore')) for word in words]
    
def use_api(message):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    rel_path = 'trained_model/%s' %('bow_model.sav')
    full_path = os.path.join(dir_path,rel_path)
    bmodel = pickle.load(open(full_path, 'rb'))

    rel_path = 'trained_model/%s' %('tfidf_model.sav')
    full_path = os.path.join(dir_path,rel_path)
    tmodel = pickle.load(open(full_path, 'rb'))

    rel_path = 'trained_model/%s' %('machine_learning.sav')
    full_path = os.path.join(dir_path,rel_path)
    mlmodel = pickle.load(open(full_path, 'rb'))

    bmsg = bmodel.transform(message)
    tmsg = tmodel.transform(bmsg)
    result = mlmodel.predict(tmsg)
    if not any([x == 1 for x in result]):
        return 'message is suspicious'
    else:
        return 'message is not suspicious'
   
if __name__ == '__main__':
    '''
    For the user of the API :
    Please pass in the  message that needs to be classified as suspicious or not-suspicious
    '''
    arg1 = sys.argv[1]
    dir_path = os.path.dirname(os.path.realpath(__file__))
    rel_path = '%s' %(arg1)
    full_path = os.path.join(dir_path,rel_path)
    print "Read the input"
    with open(full_path) as f:
        message =  f.readlines()
    message = message[0].rstrip()
    msg_list =list()
    msg_list.append(message)
    print msg_list
    print "processing the input......"
    print use_api(msg_list)
