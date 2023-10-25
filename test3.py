import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#匯入檔案
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
# data.dropna(axis=0, inplace=True)

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
data['income'] = data['income'].apply(lambda x: 1 if x.strip() == ('>50K' or '>50K.') else 0)

# 'income' 是目標變數，將其從資料中分離出來
features = data.drop('income', axis=1)  # 特徵
labels = data['income']  # 目標


#正規化
# 創建MinMaxScaler對象
regular = MinMaxScaler()
# 將數據進行最小-最大正規化
data_frame = pd.DataFrame(regular.fit_transform(data))

# 分割成訓練和測試資料（測試資料佔總資料的20%）
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 創建不同參數設定下的參數
par_max_depth = [1, 2, 3]
par_ccp_alpha = [0.001, 0.002, 0.003]

models = []
# 計算和比較不同參數設定下的分類預測正確率
train_accuracies = []
test_accuracies = []

for p in range(3):
    # clf = DecisionTreeClassifier(criterion='entropy')
    clf = DecisionTreeClassifier(max_depth=par_max_depth[p], ccp_alpha=par_ccp_alpha[p])
    clf.fit(X_train, y_train)
    models.append(clf)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

print("決策樹的分類預測正確率")
print(test_accuracies)

# 繪製決策樹
for i in range(3):
    plt.figure()
    plot_tree(models[i], filled=True, feature_names=X_train.columns, class_names=['<=50K', '>50K'])
    plt.show()
    
    