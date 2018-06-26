import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import preprocessing
train = pd.read_csv("../input/train_HK6lq50.csv")
test = pd.read_csv("../input/test_2nAIblo.csv")
sample = pd.read_csv("../input/sample_submission_vaSxamm.csv")


train["source"]= "train"
test["source"] = "test"
data=pd.concat([train, test],ignore_index=True)

# You can use keys parameter to get rid of adding new column like 
# data = pd.concat([train,test],keys=['train','test'] 
# For accessing you can use data.loc['train'], data.loc['test'] 

print(data.shape)

print(data.apply(lambda x: sum(x.isnull())))

def lable_encode_edu(x):
    switcher = {
        "High School Diploma": 1,
        "Matriculation": 2,
        "Bachelors": 3,
        "Masters": 4,
    }
    return switcher.get(x, 0)

data["education"] = data["education"].apply(lambda x: lable_encode_edu(x))

# Apply ain't that good for performace so you can go for map and fillna 
# data['education'] = data['education'].map({
#        "High School Diploma": 1,
#        "Matriculation": 2,
#        "Bachelors": 3,
#        "Masters": 4,
#    }).fillna(0)

def lable_encode_diff(x):
    switcher = {
        "easy": 1,
        "intermediate": 2,
        "hard": 3,
        "vary hard": 4,
    }
    return switcher.get(x, 0)

data["difficulty_level"] = data["difficulty_level"].apply(lambda x: lable_encode_diff(x))

# Apply ain't that good for performace so you can go for map and fillna 
# data['difficulty_level'] = data['difficulty_level'].map({
#        "easy": 1,
#        "intermediate": 2,
#        "hard": 3,
#        "vary hard": 4,
#    }).fillna(0)


dummy_fields = ["city_tier","gender","is_handicapped","program_id","program_type","test_type"] # no program_duration test_id for it's feature importance
 
for item in dummy_fields:
    dummies = pd.get_dummies(data.loc[:, item], prefix=item) 
    data = pd.concat([data, dummies], axis = 1)
    data = data.drop(item, axis =1)
    
# You can also do this rather than using a for loop
# data = pd.concat([data.drop(dummy_fiels,axis=1),pd.get_dummies(data[dummy_fields].astype(str))], axis= 1)   

def missing_flag(x):
    switcher = {
        True: 1,
    }
    return switcher.get(x, 0)

data["age_missing_flag"] = data.isnull()["age"].apply(lambda x:missing_flag(x))
data["rating_missing_flag"] = data.isnull()["trainee_engagement_rating"].apply(lambda x:missing_flag(x))

# Try to avoid apply as much as possible. 
# data["age_missing_flag"] = data["age"].isnull().astype(int) 
# data["rating_missing_flag"] = data["trainee_engagement_rating"].isnull().astype(int)

data['age'].fillna(data['age'].median(),inplace=True)
data['trainee_engagement_rating'].fillna(data['trainee_engagement_rating'].median(),inplace=True)

train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source','is_pass'],axis=1,inplace=True)

train.to_csv("../feature/train_feature.csv",index=False)
test.to_csv("../feature/test_feature.csv",index=False)

print("Feature for baseline prepared.")
