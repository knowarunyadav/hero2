# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('iris.csv')
df.set_index('Id',inplace=True)
x=np.array(df.drop(['Species'],axis=1))
y=np.array(df.Species)
from sklearn import model_selection,svm
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)
clf=svm.SVC()
fit=clf.fit(x_train,y_train)
accu=clf.score(x_test,y_test)
print(accu)

# Saving model to disk
pickle.dump(clf, open('iris.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('iris.pkl','rb'))
print(model.predict([[5.1,3.5,1.4,0.2]]))
print(model.predict([[6.7,3.0,5.2,2.3]]))
