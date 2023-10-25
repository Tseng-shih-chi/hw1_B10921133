import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#匯入檔案
from ucimlrepo import fetch_ucirepo
# fetch dataset
adult = fetch_ucirepo(id=2)
# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets
print()
# metadata
print(adult.metadata)
# variable information
print(adult.variables)

# #刪除缺失值的列
data = pd.concat([X, y], axis=1)
data.replace('?', np.nan, inplace=True)
print(data.isnull().any())
data.dropna(inplace=True)
data.dropna(axis=0, inplace=True)

#category_means = data.groupby('income')['workclass'].mean()
#data['workclass'].fillna(data['income'].map(category_means), inplace=True)
#print(data)


#標籤編碼
label_encoder = LabelEncoder()
data['workclass'] = label_encoder.fit_transform(data['workclass'])
data['education'] = label_encoder.fit_transform(data['education'])
data['marital-status'] = label_encoder.fit_transform(data['marital-status'])
data['occupation'] = label_encoder.fit_transform(data['occupation'])
data['relationship'] = label_encoder.fit_transform(data['relationship'])
data['race'] = label_encoder.fit_transform(data['race'])
data['sex'] = label_encoder.fit_transform(data['sex'])
data['native-country'] = label_encoder.fit_transform(data['native-country'])

#將目標特徵轉為0,1
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)


# 'income' 是目標變數，將其從資料中分離出來
features = data.drop('income', axis=1)  # 特徵
labels = data['income']  # 目標

#正規化
# 創建MinMaxScaler對象
regular = MinMaxScaler()
# 將數據進行最小-最大正規化
data_frame = pd.DataFrame(regular.fit_transform(data))

# 分割成訓練和測試資料（測試資料佔總資料的20%）random_stated->確保使用相同數量種子
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

id3_classifier = DecisionTreeClassifier(criterion="entropy")
id3_classifier.fit(X_train, y_train)

c45_classifier = DecisionTreeClassifier(criterion="entropy", splitter="best")
c45_classifier.fit(X_train, y_train)

c50_classifier = DecisionTreeClassifier(criterion="entropy", splitter="random")
c50_classifier.fit(X_train, y_train)

cart_classifier = DecisionTreeClassifier(criterion="gini")
cart_classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

id3_classifier = DecisionTreeClassifier(criterion="entropy")
id3_classifier.fit(X_train, y_train)

c45_classifier = DecisionTreeClassifier(criterion="entropy", splitter="best")
c45_classifier.fit(X_train, y_train)

c50_classifier = DecisionTreeClassifier(criterion="entropy", splitter="random")
c50_classifier.fit(X_train, y_train)

cart_classifier = DecisionTreeClassifier(criterion="gini")
cart_classifier.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

id3_predictions = id3_classifier.predict(X_test)
id3_accuracy = accuracy_score(y_test, id3_predictions)

c45_predictions = c45_classifier.predict(X_test)
c45_accuracy = accuracy_score(y_test, c45_predictions)

c50_predictions = c50_classifier.predict(X_test)
c50_accuracy = accuracy_score(y_test, c50_predictions)

cart_predictions = cart_classifier.predict(X_test)
cart_accuracy = accuracy_score(y_test, cart_predictions)

result_data = {
    "Actual": y_test,
    "ID3 Predictions": id3_predictions,
    "C4.5 Predictions": c45_predictions,
    "C5.0 Predictions": c50_predictions,
    "CART Predictions": cart_predictions
}

result_df = pd.DataFrame(result_data)
result_df.to_excel("classification_results.xlsx", index=False)

# 列印各演算法的分類正確率
print(f"ID3 Predictions: {id3_accuracy}")
print(f"C4.5 Predictions: {c45_accuracy}")
print(f"C5.0 Predictions: {c50_accuracy}")
print(f"CART Predictions: {cart_accuracy}")


