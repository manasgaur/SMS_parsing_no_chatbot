import pandas as pd
import pickle,csv, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import codecs,string
import Tweet_tokenizer as tt
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.utils import shuffle

wnl = WordNetLemmatizer()
def detect_language(character):
    maxchar = max(character)
    if u'\u0900' <= maxchar <= u'\u097f':
        return 'hindi'

def lemmatization(message):
    ## making all the messages of the same format ( small caps)
    message = unicode(message, 'utf-8').lower()
    words =  tt.tokenize(message)
    temp=list()
    for word in words:
        if detect_language(word) == 'hindi':
            return -1
        else:
            temp.append(word.encode('utf-8'))
    return ' '.join(word for word in temp)

def lemmatization_for_CV(message):
    message = unicode(message, 'utf-8').lower()
    words =  tt.tokenize(message)
    return [wnl.lemmatize(word.lower().encode('ascii','ignore')) for word in words]

def tfidf_BOW_model(training_dataset):
    print "---- Performing simple NLP task of text preprocessing"
    BOW_model = CountVectorizer(analyzer=lemmatization_for_CV).fit(training_dataset.message.tolist())
    BOW_of_messages = BOW_model.transform(training_dataset.message.tolist())
    tfidf = TfidfTransformer().fit(BOW_of_messages)
    tfidf_over_train = tfidf.transform(BOW_of_messages)
    print "Tf-IDF, Bag of words model created."
    return BOW_model, tfidf, tfidf_over_train

def model_training(vectors, label):
    classifier = RandomForestClassifier(n_estimators=14, random_state=15435).fit(vectors, label)
    return classifier

def model_testing(text_message, tfidf, BOW_model, classifier):
    Test_BOW_message = BOW_model.transform(text_message)
    test_tfidf_over_BOW_of_message = tfidf.transform(Test_BOW_message)
    test_prediction = classifier.predict(test_tfidf_over_BOW_of_message)
    return test_prediction

def model_pipeline(final_dataset):
    print "Creating the testing dataset\n"
    testing_dataset = final_dataset.sample(frac=0.3, replace=True)
    print "Creating the training dataset \n"
    training_dataset = pd.concat([final_dataset, testing_dataset, testing_dataset]).drop_duplicates(keep=False)
    print "Creating the Tf-IDF model and Bag of words Model"
    BOW_model, tfidf, tfidf_over_train = tfidf_BOW_model(training_dataset, vocab)
    print "Performing Cross Validation......"
    TrmsgV,ValmsgV,Trlbl,Vallbl = train_test_split(tfidf_over_train, training_dataset.label, test_size=0.33)
    print "Model training starts..."
    classifier = model_training(TrmsgV, Trlbl)
    print "Model has been trained"
    validation_set_label_prediction = classifier.predict(ValmsgV)
    training_accuracy = accuracy_score(Vallbl, validation_set_label_prediction)
    print "Testing the model"
    predictions = model_testing(testing_dataset.message, tfidf, BOW_model, classifier)
    testing_accuracy = accuracy_score(testing_dataset.label,predictions)
    print "Pipeline completed!!"
    return training_accuracy, testing_accuracy, classifier, BOW_model, tfidf

if __name__ == '__main__':
    arg1 = sys.argv[1]
    if not arg1:
        print "Please enter the name of suspect messages file. Unable to proceed."
        raise FileNotFoundError
    suspect_message_frame = pd.read_csv(arg1)
    suspect_message_frame = suspect_message_frame.sample(frac=1).reset_index(drop=True)
    ab_actual_dataset = pd.DataFrame(columns=["message", "label"])
    ab_actual_dataset.message = suspect_message_frame.message.copy(deep=True)
    ab_actual_dataset.label = suspect_message_frame.processed.copy(deep=True)
    temp = list(ab_actual_dataset.label)
    true_indices = [i for i, x in enumerate(temp) if x == True]
    false_indices = [i for i, x in enumerate(temp) if x == False]
    new_temp = np.zeros((len(temp)))
    for idx in true_indices:
        new_temp[idx] = 1
    ab_actual_dataset.label = new_temp
    print ab_actual_dataset.label.head()
    final_dataset = pd.DataFrame(columns=['message', 'label'])
    print "Data Preparation begin ..... \n"
    for index, row in ab_actual_dataset.iterrows():
        if lemmatization(row.message) == -1:
            continue
        else:
            final_dataset = final_dataset.append({'message':lemmatization(row.message), 'label':row.label},ignore_index=True)
    print "Data Prepared. Pipeline begins.... \n"
    final_dataset = shuffle(final_dataset)
    Tr_score, Te_score, created_model, bow_model, tfidf_model = model_pipeline(final_dataset)
    print "training accuracy:", Tr_score,"\t", "Testing Accuracy:", Te_score
    user_input = raw_input("Are you satisfied with the model? (y/n) (case-sensitive)")
    if user_input == 'y':
        re_input = raw_input("Do you want to save the model for future use ? (y/n) (case-sensitive) ")
        if re_input == 'y':
            model_name = 'machine_learning.sav'
            bowmodel_name = 'bow_model.sav'
            tfidfmodel_name = 'tfidf_model.sav'
            print " Saving the machine learning model ...."
            pickle.dump(created_model, open('trained_model/%s' %(model_name), 'wb'))
            print " Machine learning model saved. Saving BOW and tfidf model ....."
            pickle.dump(bow_model, open('trained_model/%s' %(bowmodel_name), 'wb'))
            pickle.dump(tfidf_model, open('trained_model/%s' %(tfidfmodel_name),'wb'))
            print " BOW and Tf-idf model have been saved. Now you can use them using \"using pickle.load()\""
        else:
            warnings.formatwarning = custom_warning_message
            warnings.warn("model deleted!!!!")
    else:
        warnings.formatwarning = custom_warning_message
        warnings.warn("model deleted!!!!") 