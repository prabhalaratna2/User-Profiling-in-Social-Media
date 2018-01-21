import os
import sys
import pandas as pd
import numpy as np
import nltk
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

class Emotions_By_Status():
    def run(testdf, test_data_directory):

     print("Predicting emotions from Status...")
     training_profile_data_emotions = "/data/training/profile/profile.csv"
     df_profile_emotions = pd.read_csv(training_profile_data_emotions)
     #test_data_emotions = test_data_directory + '/profile/profile.csv'
     #testdf_profile_emotions = pd.read_csv(test_data_emotions)

     df_LIWCFeature_training = pd.read_csv(r'/data/training/LIWC/LIWC.csv')
     df_LIWCFeature_test=pd.read_csv(test_data_directory+'/LIWC/LIWC.csv')

     df_LIWCFeature_training.columns = [x.lower() for x in df_LIWCFeature_training.columns]
     df_LIWCFeature_merge_profile_training = pd.merge(df_LIWCFeature_training,df_profile_emotions,on='userid')
     df_LIWCFeature_merge_profile_training.drop(['Unnamed: 0','age','gender'],axis=1, inplace=True)

     df_LIWCFeature_test.columns = [x.lower() for x in df_LIWCFeature_test.columns]
     df_LIWCFeature_merge_profile_test = pd.merge(df_LIWCFeature_test,testdf,on='userid')
     df_LIWCFeature_merge_profile_test.drop(['Unnamed: 0','age','gender'],axis=1, inplace=True)

     # Preparing the train and test data
     big5 = ['ope','ext','con','agr','neu']
     LIWC_features = [x for x in df_LIWCFeature_merge_profile_training.columns.tolist()[:] if not x in big5]
     LIWC_features.remove('userid')
     X_train_emotions = df_LIWCFeature_merge_profile_training[LIWC_features]
     y_train_ope = df_LIWCFeature_merge_profile_training.ope #selecting ope as the target
     y_train_con = df_LIWCFeature_merge_profile_training.con #selecting con as the target
     y_train_ext = df_LIWCFeature_merge_profile_training.ext #selecting extrovert as the target
     y_train_agr = df_LIWCFeature_merge_profile_training.agr #selecting agr as the target
     y_train_neu = df_LIWCFeature_merge_profile_training.neu #selecting neurotic as the target

     LIWC_features = [x for x in df_LIWCFeature_merge_profile_test.columns.tolist()[:] if not x in big5]
     LIWC_features.remove('userid')
     X_test_emotions = df_LIWCFeature_merge_profile_test[LIWC_features]

     # Training and evaluating a linear regression model
     linreg_ope = LinearRegression()
     linreg_con = LinearRegression()
     linreg_ext = LinearRegression()
     linreg_agr = LinearRegression()
     linreg_neu = LinearRegression()

     linreg_ope.fit(X_train_emotions,y_train_ope)
     linreg_con.fit(X_train_emotions,y_train_con)
     linreg_ext.fit(X_train_emotions,y_train_ext)
     linreg_agr.fit(X_train_emotions,y_train_agr)
     linreg_neu.fit(X_train_emotions,y_train_neu)
     # Evaluating the model
     y_predict_emotions_ope = linreg_ope.predict(X_test_emotions)
     y_predict_emotions_con = linreg_con.predict(X_test_emotions)
     y_predict_emotions_ext = linreg_ext.predict(X_test_emotions)
     y_predict_emotions_agr = linreg_agr.predict(X_test_emotions)
     y_predict_emotions_neu = linreg_neu.predict(X_test_emotions)

     print("Average of Openness Emotions: ",np.mean(y_predict_emotions_ope))
     print("Average of Conscientiousness Emotions: ",np.mean(y_predict_emotions_con))
     print("Average of Extroversion Emotions: ",np.mean(y_predict_emotions_ext))
     print("Average of Agreeableness Emotions: ",np.mean(y_predict_emotions_agr))
     print("Average of Emotional Stability: ",np.mean(y_predict_emotions_neu))

     userid_to_emotion_ope_dictionary = dict()
     userid_to_emotion_con_dictionary = dict()
     userid_to_emotion_ext_dictionary = dict()
     userid_to_emotion_agr_dictionary = dict()
     userid_to_emotion_neu_dictionary = dict()
     userid_to_emotion_dictionary = dict()

     for index, row in testdf.iterrows():
        userid = getattr(row , 'userid')
        userid_to_emotion_ope_dictionary[userid]  = y_predict_emotions_ope[index]
        userid_to_emotion_con_dictionary[userid]  = y_predict_emotions_con[index]
        userid_to_emotion_ext_dictionary[userid]  = y_predict_emotions_ext[index]
        userid_to_emotion_agr_dictionary[userid]  = y_predict_emotions_agr[index]
        userid_to_emotion_neu_dictionary[userid]  = y_predict_emotions_neu[index]
        userid_to_emotion_dictionary[userid] = [y_predict_emotions_ope[index], y_predict_emotions_con[index], y_predict_emotions_ext[index], y_predict_emotions_agr[index], y_predict_emotions_neu[index]]
     return userid_to_emotion_dictionary #userid_to_emotion_ope_dictionary,userid_to_emotion_con_dictionary,userid_to_emotion_ext_dictionary,userid_to_emotion_agr_dictionary,userid_to_emotion_neu_dictionary

