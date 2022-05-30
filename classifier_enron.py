<<<<<<< HEAD
# import libraries


# Helper functions to create dictionary and extract features from the corpus for model development
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    # Code for non-word removal
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000) 
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    return features_matrix

# Use the above two functions to load dataset, create training, test splits for model development


# Train SVM model


# Test SVM model on unseen emails (test_matrix created from extract_features function)


# Save your model as .sav file to upload with your submission
=======
# import libraries
import os
import numpy as np
from collections import Counter
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

# Helper functions to create dictionary and extract features from the corpus for model development
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []  
    for file in emails:
      directory = [os.path.join(file, f) for f in os.listdir(file)] 
      for dr in directory:
        emails = [os.path.join(dr, f) for f in os.listdir(dr)] #   
      for mail in emails:    
        with open(mail, errors = "ignore") as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    
    dictionary = Counter(all_words)
    # Code for non-word removal
    list_to_remove = dictionary.keys()
    for item in list_to_remove:
        if item.isalpha() == False: 
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(3000) 
    return dictionary

def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    train_labels = np.zeros(33716)#total emails
    for fil in files:
      dirs = [os.path.join(fil, f) for f in os.listdir(fil)]
      for d in dirs:
        emails = [os.path.join(d, f) for f in os.listdir(d)] #change 'emails'
        for mail in emails:
         with open(fil, errors = "ignore") as fi:
             all_words = []
             for i,line in enumerate(fi):
                 if i == 2:
                     words = line.split()
                     all_words += words
                     for word in all_words:
                         wordID = 0
             for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = all_words.count(word)
                  train_labels[docID] = int(mail.split(".")[-2] == 'spam')
        docID = docID + 1     
    return features_matrix, train_labels

# Use the above two functions to load dataset, create training, test splits for model development
train_dir = r'C:\Users\corbi\Downloads\CSE 5120 emailClassification\enron-spam\enron-spam'
dictionary = make_Dictionary(train_dir)
features, labels = extract_features(train_dir)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.40)


train_matrix = extract_features(train_dir)

# Train SVM model
#model1 = LinearSVC()
model1 = SVC()
model1.fit(X_train, y_train)

# Test SVM model on unseen emails (test_matrix created from extract_features function)
result = model.predict(X_test)

print(confusion_matrix(y_test, result))
print(accuracy_score(y_test, result))
print('Accuracy:'"{:.2%}".format(accuracy_score(y_test,result)))

# Save your model as .sav file to upload with your submission
saved_model = "emailClassifier_enron.sav"
pickle.dump(model1, open(saved_model, 'wb'))
>>>>>>> 51c073b (Added Update 5/30)
