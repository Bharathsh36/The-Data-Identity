
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing


train = pd.read_csv("../input/train_HK6lq50.csv")
test = pd.read_csv("../input/test_2nAIblo.csv")
sample = pd.read_csv("../input/sample_submission_vaSxamm.csv")


print(train.shape,test.shape)

train["source"]= "train"
test["source"] = "test"
data=pd.concat([train, test],ignore_index=True)
print(data.shape)

print(data.apply(lambda x: sum(x.isnull())))


for item in data.columns:
    print (item)
    print (data[item].value_counts())
    print ("\n")


# LableEncoder
# difficulty_level,education(No Qualification first),


def lable_encode_edu(x):
    switcher = {
        "High School Diploma": 1,
        "Matriculation": 2,
        "Bachelors": 3,
        "Masters": 4,
    }
    return switcher.get(x, 0)


data["education"] = data["education"].apply(lambda x: lable_encode_edu(x))

def lable_encode_diff(x):
    switcher = {
        "easy": 1,
        "intermediate": 2,
        "hard": 3,
        "vary hard": 4,
    }
    return switcher.get(x, 0)


data["difficulty_level"] = data["difficulty_level"].apply(lambda x: lable_encode_diff(x))

# get_dummies
# city_tier,gender,is_handicapped,program_duration,program_id,program_type,test_id,test_type,trainee_id(too many to do),

dummy_fields = ["city_tier","gender","is_handicapped","program_id","program_type","test_type","program_duration","test_id"] 

for item in dummy_fields:
    dummies = pd.get_dummies(data.loc[:, item], prefix=item) 
    data = pd.concat([data, dummies], axis = 1)
    data = data.drop(item, axis =1)

print(data.shape)

#missing age,trainee_engagement_rating      

def missing_flag(x):
    switcher = {
        True: 1,
    }
    return switcher.get(x, 0)

data["age_missing_flag"] = data.isnull()["age"].apply(lambda x:missing_flag(x))
data["rating_missing_flag"] = data.isnull()["trainee_engagement_rating"].apply(lambda x:missing_flag(x))


data['age'].fillna(data['age'].median(),inplace=True)
data['trainee_engagement_rating'].fillna(data['trainee_engagement_rating'].median(),inplace=True)


#get_dummies plus 
#age,trainee_engagement_rating,age_missing_flag,rating_missing_flag,difficulty_level,education,total_programs_enrolled,trainee_engagement_rating
dummy_fields_plus = ["age","trainee_engagement_rating","age_missing_flag","rating_missing_flag","difficulty_level","education","total_programs_enrolled"] 

for item in dummy_fields_plus:
    dummies = pd.get_dummies(data.loc[:, item], prefix=item) 
    data = pd.concat([data, dummies], axis = 1)
    data = data.drop(item, axis =1)

print(data.shape)

# TODO: trainee_id box
data.drop('trainee_id',axis=1,inplace=True)
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source','is_pass'],axis=1,inplace=True)

train.to_csv("../feature/train_nn.csv",index=False)
test.to_csv("../feature/test_nn.csv",index=False)

print("Feature_nn.py Done.")