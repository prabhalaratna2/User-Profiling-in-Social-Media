import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics


class Gender_By_Status():
    def run(train_profile_df, test_profile_df, test_data_directory):

        print("Predicting Gender from status...")

        for index, row in train_profile_df.iterrows():
            data_userid = row['userid']
            userid_to_txt = open('/data/training/text/' + data_userid + '.txt', 'r', errors='ignore')
            data_status = userid_to_txt.read()
            train_profile_df.set_value(index, 'Status', data_status)
            userid_to_txt.close()
        
        for index, row in test_profile_df.iterrows():
            data_userid = row['userid']
            userid_to_txt = open(test_data_directory + '/text/' + data_userid + '.txt', 'r', errors='ignore')
            data_status = userid_to_txt.read()
            test_profile_df.set_value(index, 'Status', data_status)
            userid_to_txt.close()

       ########################### Getting training and test data#######################

        data_FBUsers_train = train_profile_df.loc[:, ['userid', 'gender', 'Status']]
        data_FBUsers_test = test_profile_df.loc[:, ['userid', 'gender', 'Status']]

        ################################################################################

        test_data = data_FBUsers_test.loc[np.arange(len(data_FBUsers_test)), :]
        train_data = data_FBUsers_train.loc[np.arange(len(data_FBUsers_train)), :]

        ##################################################################################
        
        text_feature = ['Status']

        X_gender = train_data[text_feature]
        Y_gender = train_data.gender


        SGDModel_gender = SGDClassifier(shuffle = True)
        count_vect=CountVectorizer()
        Tf_Idf_Transformer = TfidfTransformer()
        accuracy = 0
        gender_Kfold = KFold(n_splits= 10, shuffle = True)
        for training_index, test_index in gender_Kfold.split(X_gender, Y_gender) :

           X_train_gender, X_test_gender =  X_gender.loc[training_index,], X_gender.loc[test_index,]
           Y_train_gender, Y_test_gender =  Y_gender.loc[training_index,], Y_gender.loc[test_index,]

           gender_training_count_vect = count_vect.fit_transform(X_train_gender.Status)
           gender_training_Tf_Idf_Transformer = Tf_Idf_Transformer.fit_transform(gender_training_count_vect)

           SGDModel_gender.fit(gender_training_Tf_Idf_Transformer,Y_train_gender)

           gender_test_count_vect = count_vect.transform(X_test_gender.Status)
           gender_test_Tf_Idf_Transformer = Tf_Idf_Transformer.transform(gender_test_count_vect)

           gender_predicted = SGDModel_gender.predict(gender_test_Tf_Idf_Transformer)
           accuracy+=accuracy_score(Y_test_gender,gender_predicted)
        gender_test_count_vect = count_vect.transform(test_data.Status)
        gender_test_Tf_Idf_Transformer = Tf_Idf_Transformer.transform(gender_test_count_vect)

        gender_predicted = SGDModel_gender.predict(gender_test_Tf_Idf_Transformer)
        y_predicted = np.int_(gender_predicted)
        userId_to_GenderByStatus = dict()
        for index, row in test_data.iterrows():
           userid = getattr(row , 'userid')
           #print(userid)
           #print(y_predicted[index])
           userId_to_GenderByStatus[userid] = y_predicted[index]
        return userId_to_GenderByStatus

