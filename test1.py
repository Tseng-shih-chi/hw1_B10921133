import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

#匯入檔案
#pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
# fetch dataset
adult = fetch_ucirepo(id=2)
# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets
# metadata
#print(adult.metadata)
# variable information
#print(adult.variables)


# #刪除缺失值的列
data = pd.concat([X, y], axis=1)
data.replace('?', np.nan, inplace=True)
# print(data.isnull().any())
data.dropna(inplace=True)
# print(data)


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

# 創建CART決策樹分類器
clf = DecisionTreeClassifier()
# 使用訓練資料進行訓練
clf.fit(X_train, y_train)

# 預測訓練資料和測試資料
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# 計算分類正確率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("訓練資料分類正確率:", train_accuracy)
print("測試資料分類正確率:", test_accuracy)