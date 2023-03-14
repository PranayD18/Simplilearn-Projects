# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#  %matplotlib inline

columnsUsers=['UserID','Gender','Age','OccupationNo','Zip-code']
columnRatings=['UserID','MovieID','Rating','Timestamp']
columnMovies=['MovieID','Title','Genres']

df_users=pd.read_table(r'C:\Users\user\Desktop\Learning\Jupyter Workspace\Project\Data science with Python 1\users.dat',
                        sep='::',header=None,
                        names=columnsUsers)


df_ratings=pd.read_table(r'C:\Users\user\Desktop\Learning\Jupyter Workspace\Project\Data science with Python 1\ratings.dat',
                        sep='::',header=None,
                        names=columnRatings)


df_movies=pd.read_table(r'C:\Users\user\Desktop\Learning\Jupyter Workspace\Project\Data science with Python 1\movies.dat',
                        sep='::',header=None,
                        names=columnMovies)

df_movies['Action']=pd.np.where(df_movies.Genres.str.contains("Action"), "Y","N")
df_movies['Adventure']=pd.np.where(df_movies.Genres.str.contains("Adventure"), "Y","N")
df_movies['Animation']=pd.np.where(df_movies.Genres.str.contains("Animation"), "Y","N")
df_movies['Children']=pd.np.where(df_movies.Genres.str.contains("Children"), "Y","N")
df_movies['Comedy']=pd.np.where(df_movies.Genres.str.contains("Comedy"), "Y","N")
df_movies['Crime']=pd.np.where(df_movies.Genres.str.contains("Crime"), "Y","N")
df_movies['Documentary']=pd.np.where(df_movies.Genres.str.contains("Documentary"), "Y","N")
df_movies['Drama']=pd.np.where(df_movies.Genres.str.contains("Drama"), "Y","N")
df_movies['Fantasy']=pd.np.where(df_movies.Genres.str.contains("Fantasy"), "Y","N")
df_movies['Film-Noir']=pd.np.where(df_movies.Genres.str.contains("Film-Noir"), "Y","N")
df_movies['Horror']=pd.np.where(df_movies.Genres.str.contains("Horror"), "Y","N")
df_movies['Musical']=pd.np.where(df_movies.Genres.str.contains("Musical"), "Y","N")
df_movies['Mystery']=pd.np.where(df_movies.Genres.str.contains("Mystery"), "Y","N")
df_movies['Romance']=pd.np.where(df_movies.Genres.str.contains("Romance"), "Y","N")
df_movies['Sci-Fi']=pd.np.where(df_movies.Genres.str.contains("Sci-Fi"), "Y","N")
df_movies['Thriller']=pd.np.where(df_movies.Genres.str.contains("Thriller"), "Y","N")
df_movies['War']=pd.np.where(df_movies.Genres.str.contains("War"), "Y","N")
df_movies['Western']=pd.np.where(df_movies.Genres.str.contains("Western"), "Y","N")


idx=['MovieID','Title','Genres']
# Then pivot the dataset based on this multi-level index 
multi_indexed_df = df_movies.set_index(idx)

stacked_df = multi_indexed_df.stack(dropna=False)

long_df = stacked_df.reset_index()

df_new = long_df.rename(columns={'MovieID':'MovieID','Title':'Title','Genres':'GenresGroup','level_3':'Genre',0:'Value'})

df_movies=df_new[df_new['Value']=='Y']

df_users['AgeGroup'] = pd.cut(x=df_users['Age'],
                              bins=[0,1,18,25,35,45,50,56],labels=['Below 18','18-24','25-34','35-44','45-54','50-55','56+'])



# initialize list of lists 
data = { 'Occupation' : ['other','academic/educator','artist','clerical/admin' , 'college/grad student' , 'customer service' , 'doctor/health care',
        'executive/managerial' , 'farmer' , 'homemaker' , 'K-12 student' , 'lawyer' , 'programmer' , 'retired' , 'sales/marketing' , 'scientist' ,
        'self-employed' , 'technician/engineer' , 'tradesman/craftsman' , 'unemployed' , 'writer'],
        
        'OccupationNo': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}

# Create the pandas DataFrame 
df_Occupation = pd.DataFrame(data)

df_users=pd.merge(df_users,df_Occupation,on='OccupationNo',how='left')

df1=pd.merge(df_ratings,df_users,on='UserID',how='left') 
df2=pd.merge(df1,df_movies,on='MovieID',how='left')

df_MasterData=df2[['MovieID','Title','UserID','Age','AgeGroup','Gender','Occupation','Rating']]


#sns.distplot(df_MasterData.Age)

#df_MasterData['Age'].hist()

AgeGroup=df_MasterData['Age'].value_counts()

df_TS=df_MasterData[df_MasterData['Title']=='Toy Story (1995)']
ToyStoryRating=df_TS['Rating'].value_counts()

df4=df_MasterData.drop_duplicates(subset=['MovieID','Title','Rating'])
df_top25=df4.nlargest(25, ['Rating']) 

print(df_top25[['MovieID','Title','Rating']])

df3=df2[['MovieID','Age','Rating']]
X=df3.iloc[:,:2]
y=df3.iloc[:,2:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
from sklearn import metrics

actual = y_test
predicted = model.predict(X_test)
print(accuracy_score(actual,predicted))

print(metrics.classification_report(actual,predicted))



