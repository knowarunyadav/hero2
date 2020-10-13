# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('titanic.csv')
df.columns = map(str.lower,df.columns)
df.set_index("passengerid",inplace=True)
df.age = df.age.fillna(df.age.mean())

for i in df.columns:

    if df[i].dtype != np.int64 and df[i].dtype != np.float64:

        column = df[i].values.tolist()
        column = list(set(column))
        dis = {}
        count = 0
        def value(val):
            return dis[val]

        for j in column:

            dis[j] = count
            count +=1

        df[i] = list(map(value,df[i]))

df.info()

x = np.array(df.drop(['survived'],axis=1))
y = df.iloc[:,0].values
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)
clf=LogisticRegression()
clf.fit(x_train,y_train)
accu=clf.score(x_test,y_test)
print(accu)

# Saving model to disk
pickle.dump(clf, open('titanic.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('titanic.pkl','rb'))
print(model.predict([[  3., 229.,   0.,  47.,   1.,   0.,  14.,   7.,   0.,   0.]]))
print(model.predict([[  3.    , 274.    ,   0.    ,  30.    ,   0.    ,   0.    ,
       313.    ,   7.6292,   0.    ,   2.    ]]))
