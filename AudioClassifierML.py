import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# import numpy as np
# import matplotlib as plt
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectFromModel

#df = pd.read_csv(r"./italian_dataset.csv")
df_train = pd.read_csv(r"features/train_features.csv")
df_validation = pd.read_csv(r"features/validation_features.csv")
df_test = pd.read_csv(r"features/test_features.csv")

# print(df_validation)

#  drop the first column. It's the voiceID
df_train.drop('voiceID', inplace=True, axis=1)
df_validation.drop('voiceID', inplace=True, axis=1)
df_test.drop('voiceID', inplace=True, axis=1)


#  separate dependent and independent variable
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_validation = df_validation.iloc[:, :-1].values
y_validation = df_validation.iloc[:, -1].values

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

# Fit classifier to the Training set
model_knn = KNeighborsClassifier(n_neighbors=10)
model_knn.fit(X_train, y_train)

model_svm = svm.SVC()
model_svm.fit(X_train, y_train)

model_rf = RandomForestClassifier(max_depth=100, random_state=0)
model_rf.fit(X_train, y_train)

model_gb = GradientBoostingClassifier(random_state=0)
model_gb.fit(X_train, y_train)

#predict
y_pred_knn = model_knn.predict(X_validation)
y_pred_svm = model_svm.predict(X_validation)
y_pred_rf = model_rf.predict(X_validation)
y_pred_gb = model_gb.predict(X_validation)

conf_matrix_knn = confusion_matrix(y_validation, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_validation, y_pred_svm)
conf_matrix_rf = confusion_matrix(y_validation, y_pred_rf)
conf_matrix_gb = confusion_matrix(y_validation, y_pred_gb)
print("KNN Confusion matrix\n", conf_matrix_knn)
print("\nSVM Confusion matrix\n", conf_matrix_svm)
print("\nRF Confusion matrix\n", conf_matrix_rf)
print("\nGB Confusion matrix\n", conf_matrix_gb)

accuracy_knn = ((conf_matrix_knn[0, 0] + conf_matrix_knn[1, 1]) / (
        conf_matrix_knn[0, 0] + conf_matrix_knn[0, 1] + conf_matrix_knn[1, 0] + conf_matrix_knn[1, 1])) * 100
accuracy_svm = ((conf_matrix_svm[0, 0] + conf_matrix_svm[1, 1]) / (
        conf_matrix_svm[0, 0] + conf_matrix_svm[0, 1] + conf_matrix_svm[1, 0] + conf_matrix_svm[1, 1])) * 100
accuracy_rf = ((conf_matrix_rf[0, 0] + conf_matrix_rf[1, 1]) / (
        conf_matrix_rf[0, 0] + conf_matrix_rf[0, 1] + conf_matrix_rf[1, 0] + conf_matrix_rf[1, 1])) * 100
accuracy_gb = ((conf_matrix_gb[0, 0] + conf_matrix_gb[1, 1]) / (
        conf_matrix_gb[0, 0] + conf_matrix_gb[0, 1] + conf_matrix_gb[1, 0] + conf_matrix_gb[1, 1])) * 100

print("\nKNN accuracy: ",  accuracy_knn)
print("SVM accuracy: ", accuracy_svm)
print("RF accuracy: ", accuracy_rf)
print("GB accuracy: ", accuracy_gb)