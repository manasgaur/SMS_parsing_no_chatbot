README

A. Installation of Libraries: Pre-requisite. The system should have Python 2.7.X installed.

cd to -> Albert_Pinto (in terminal or command line)
sudo -H pip install -r requirements.txt ( -H : optional)

B. Description of the contents of the zip file :

1. All .csv files are the data on which the model was trained.
2. All .py files are the source code that will be used in the future for re-training of the model and use of the API.
3. Inside the "trained_model" folder, there are three models :
bow_model.sav: this is the bag of words model created using the vocabulary of positive and valid sender keywords.
tfidf_model.sav: This model creates the word frequency into some real number that can be interpreted by the model. 
machine_learning.sav : we pass the vector from tf-IDF into this machine learning model which will predict whether the message is suspicious or not
4. Tweet_tokenizer: this is a special python source code developed for tokenizing small and informal text (e.g. tweet). This was created considering that SMS and tweets are same in structure.
5. Vocab.txt: this is a text file containing the words used as vocabulary for the model training. As such this file is of no use, it is there just for visual understanding.
6. test_message.txt: this file is important. It contains the same message that will be read by "ab_api.py" when you pass this text file.

C: Execution Instruction:

1. If you want to use the API, without retraining the model : 

python ab_api.py test_message.txt

2. If you want to retrain the model: 

a . First, delete all the models inside the folder " trained _model"  using the following command

rm -rf trained_model/*

b. After deleting all the models. type following command on the terminal :

python ab_code_with_keywords_to_be_used.py suspect-keywords-20171017-131042.csv suspect-messages-20171017-131005.csv 

This program will take you on a tour of model training, testing and storing. Follow the instruction on the terminal.

D. The utility of this project :

1. The zip file that is being sent by Manas Gaur already has pre-trained models that can be used with API: ab_api.py

2. If you want to add new words to the vocabulary, add all the new keywords inside 
"suspect-keywords-20171017-131042.csv", specifying all the values under their respective fields. The re-train the model following steps mentioned in C.2.b.

3. In its current state, the API reads one input (message) from the text file. For making it read multiple messages, it is easy: just make call to use_api(message) in ab_api.py in a for-loop or while loop

4. If a new message is tested on the model, make sure to add that message and the response of the model or human response (it the model response is not appropriate) under python "suspect-messages-20171017-131005.csv". Again re-train the model as stated in C.2.b
This will help the model to understand where it went wrong and will learn alongside human intervention (if required).

Happy to help.

Thanks
Manas Gaur