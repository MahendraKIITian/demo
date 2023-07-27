import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
import xgboost
import model_history
from model_history import ModelVersioning
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("C:/Users/VQ589HN/Downloads/advertising.csv")
data.columns = data.columns.map(lambda row: "_".join(row.lower().split(" ")))

X = data[['daily_time_spent_on_site', 'age' , 'area_income' ,'daily_internet_usage'  ,'male']]
y=data['clicked_on_ad']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)

model = xgboost.XGBClassifier().fit(X_train, y_train)
clf.predict(X[:2, :])
predict = model.predict(X_test)

m1 = ModelVersioning(model)
df_model = m1.Model_Artifacts()
print(df_model.head(10))



X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)


clf.predict_proba(X[:2, :])



