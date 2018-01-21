import sys
import os
import pandas as pd
import random
import numpy as np

# Take these values from command line
from Gender_By_Likes import Gender_By_Likes
#from Age_By_Likes import Age_By_Likes
from Emotions_By_Status import Emotions_By_Status
from Gender_By_Status import Gender_By_Status
from Gender_By_Image import Gender_By_Image
from Age_By_Status import Age_By_Status
#from personalities_by_Likes import personalities_by_Likes

if sys.argv.__len__() != 3:
    print("ERROR: please specify test data directory and output directory in command line arguments")
    exit(-1)


test_data_directory = sys.argv[1]
output_directory = sys.argv[2]

# creating a directory named output_directory
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# Use actual location of the training data CSV file
training_data = "/data/training/profile/profile.csv"
df = pd.read_csv(training_data)

test_data = test_data_directory + '/profile/profile.csv'
testdf = pd.read_csv(test_data)


data_gender = df.loc[:, ['gender']]
data_Age = df.loc[:, ['age']]
data_Personality = df.loc[:, ['ope', 'con', 'ext', 'agr', 'neu']]

df_likes = pd.read_csv(r'/data/training/relation/relation.csv')
df_likes_test = pd.read_csv(test_data_directory + '/relation/relation.csv')

#Getting userid to array of emotions dictionary predicted from user likes
#userid_to_emotions_dictionary = personalities_by_Likes.run(df, df_likes, testdf, df_likes_test)

#Getting userid to gender dictionary predicted from user images
userId_to_GenderByImage = Gender_By_Image.run(testdf, test_data_directory)

#Getting userid to emotions dictionary predicted from user status
userid_to_emotions_dictionary = Emotions_By_Status.run(testdf, test_data_directory)

#Getting userid to gender dictionary predicted from user status
userId_to_GenderByStatus = Gender_By_Status.run(df, testdf, test_data_directory)

#Getting userid to gender dictionary predicted from user status
userId_to_AgeByStatus = Age_By_Status.run(df, testdf, test_data_directory)

#Getting userid to age dictionary predicted from user likes
#userId_to_AgeByLikes = Age_By_Likes.run(df,df_likes,testdf,df_likes_test)

#Getting userid to gender dictionary predicted from user likes
userId_to_GenderByLikes = Gender_By_Likes.run(df,df_likes,testdf,df_likes_test)


print("Saving output")

count_gender_match = 0
count_gender_mismatch = 0

# Generating XML files

correct_count = 0
wrong_count = 0

for row in testdf.loc[:, ['userid']].iterrows():
    userid = getattr(row[1], 'userid')
    gender_image = userId_to_GenderByImage[userid]
    gender_likes = userId_to_GenderByLikes[userid]
    gender_status = userId_to_GenderByStatus[userid]


    age_status = userId_to_AgeByStatus[userid]
    age_grp = age_status

    #age_likes = userId_to_AgeByLikes[userid]

    #gender_likes = userId_to_GenderByLikes[userid]
    #age_grp = age_likes


    #final_gender = gender_status
    #final_gender = gender_image
    #final_gender = gender_likes
    #final_gender=gender_status
    if (gender_status == 1.0 and gender_image == 1.0 and gender_likes ==0.0):
    	final_gender = gender_image
    elif (gender_status == 0.0 and gender_image == 1.0 and gender_likes ==1.0):
        final_gender = gender_image
    elif (gender_status == 1.0 and gender_image == 0.0 and gender_likes ==1.0):
        final_gender = gender_likes
    elif (gender_status == 0.0 and gender_image == 0.0 and gender_likes ==1.0):
        final_gender = gender_image
    elif (gender_status == 0.0 and gender_image == 1.0 and gender_likes ==0.0):
        final_gender = gender_status
    elif (gender_status == 1.0 and gender_image == 0.0 and gender_likes ==0.0):
        final_gender = gender_image
    else:
        final_gender = gender_status






    #ope,con,ext,agr,neu = (userId_to_EmotionsByStatus[userid])
    emotions_array_status =  userid_to_emotions_dictionary[userid]
    ope = emotions_array_status[0]
    con = emotions_array_status[1]
    ext = emotions_array_status[2]
    agr = emotions_array_status[3]
    neu = emotions_array_status[4]

    #emotions_array = userid_to_emotions_dictionary[userid]
    #ope = emotions_array[1]
    #con = emotions_array[2]
    #ext = emotions_array[3]
    #agr = emotions_array[4]
    #neu = emotions_array[5]
    # Use this when test data doesn't contain labels
    xml = '<user id="{0}"\nage_group="{1}"\ngender="{2}"\nextrovert="{5}"\nneurotic="{7}"\nagreeable="{6}"\nconscientious="{4}"\nopen="{3}"\n/>'.format(
        userid, age_grp, final_gender, ope, con, ext, agr, neu)
    text_file = open('{0}/{1}.xml'.format(output_directory, userid), "w")

    text_file.write(xml)
    text_file.close()

#print("Gender match: %d" % count_gender_match)
#print("Gender mismatch: %d" % count_gender_mismatch)

# print("Accuracy after combination: %f" % ((correct_count)/(correct_count+wrong_count)))
