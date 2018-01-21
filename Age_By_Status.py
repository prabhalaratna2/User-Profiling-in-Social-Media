import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

class Age_By_Status():
    def run(train_profile_df, test_profile_df, test_data_directory):

        print("Predicting Age from status...")
        #print(train_profile_df['age'])

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

        #train_profile_df['age_grp'] = '50_xx'
        train_profile_df.loc[(train_profile_df['age'])< 25,'age_grp'] = 'XX_24'
        train_profile_df.loc[(train_profile_df['age'] >=25) & (train_profile_df['age'] < 35), 'age_grp'] = "25_34"
        train_profile_df.loc[(train_profile_df['age'] >= 35) & (train_profile_df['age'] < 50), 'age_grp'] = "35_49"
        train_profile_df.loc[(train_profile_df['age'] >= 50), 'age_grp'] = '50_XX'
  
        ################################ Getting training and test data###############################
        data_FBUsers_train = train_profile_df.loc[:, ['userid', 'Status', 'age_grp']]
        data_FBUsers_test = test_profile_df.loc[:, ['userid', 'Status', 'age_grp']]
       
        test_data = data_FBUsers_test.loc[np.arange(len(data_FBUsers_test)), :]
        train_data = data_FBUsers_train.loc[np.arange(len(data_FBUsers_train)), :]
   

        ############################ Training and testing data using Kfold############################
        text_feature = ['Status']
        X_age = train_data[text_feature]
        Y_age = train_data.age_grp

        SGDModel_age = SGDClassifier(shuffle = True)
        count_vect_age=CountVectorizer()
        Tf_Idf_Transformer_age = TfidfTransformer()
        accuracy = 0
        age_Kfold = KFold(n_splits= 10, shuffle = True)
        for training_index, test_index in age_Kfold.split(X_age, Y_age) :

           X_train_age, X_test_age =  X_age.loc[training_index,], X_age.loc[test_index,]
           Y_train_age, Y_test_age =  Y_age.loc[training_index,], Y_age.loc[test_index,]

           age_training_count_vect = count_vect_age.fit_transform(X_train_age.Status)
           age_training_Tf_Idf_Transformer = Tf_Idf_Transformer_age.fit_transform(age_training_count_vect)

           SGDModel_age.fit(age_training_Tf_Idf_Transformer,Y_train_age)

           age_test_count_vect = count_vect_age.transform(X_test_age.Status)
           age_test_Tf_Idf_Transformer = Tf_Idf_Transformer_age.transform(age_test_count_vect)

           age_predicted = SGDModel_age.predict(age_test_Tf_Idf_Transformer)
           accuracy+=accuracy_score(Y_test_age,age_predicted)
        age_test_count_vect = count_vect_age.transform(test_data.Status)
        age_test_Tf_Idf_Transformer = Tf_Idf_Transformer_age.transform(age_test_count_vect)
        age_predicted = SGDModel_age.predict(age_test_Tf_Idf_Transformer)

        y_predicted = np.string_(age_predicted)
        userId_to_AgeByStatus = dict()
        for index, row in test_data.iterrows():
           userid = getattr(row , 'userid')
           #print(userid)
           #print(age_predicted[index])
           userId_to_AgeByStatus[userid] = age_predicted[index]

        return userId_to_AgeByStatus
