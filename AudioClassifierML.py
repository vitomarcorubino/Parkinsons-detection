import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Set the experiment to 1, 2, 3 or 4
# 1: vowels only shuffled and split 70/30
# 2: vowels only people separated 70/30
# 3: all sounds people separated 70/30
# 4: all sounds people separated 70/30 with no vowels
experiment = 1

if experiment == 1:
    df = pd.read_csv("features/ML/vowels/features_vowels.csv")

    #drop the first column. It's the voiceID
    df.drop('voiceID', inplace = True, axis = 1)

    #separate dependent and independent variable
    X = df.iloc[:, :-1]
    df_X = df.iloc[:, :-1].values
    df_Y = df.iloc[:,-1].values

    # Split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.3, random_state = 123)
else:
    if experiment == 2:
        df_train = pd.read_csv(r"features/ML/vowelsPeople7030/train_features_split3.csv")
        df_test = pd.read_csv(r"features/ML/vowelsPeople7030/test_features_split3.csv")
    else:
        if experiment == 3:
            df_train = pd.read_csv(r"features/ML/people7030/train_features_split3.csv")
            df_test = pd.read_csv(r"features/ML/people7030/test_features_split3.csv")
        else:
            if experiment == 4:
                df_train = pd.read_csv(r"features/ML/people7030_noVowels/train_features_split3.csv")
                df_test = pd.read_csv(r"features/ML/people7030_noVowels/test_features_split3.csv")

    #  drop the first column. It's the voiceID
    df_train.drop('voiceID', inplace=True, axis=1)
    df_test.drop('voiceID', inplace=True, axis=1)

    #  separate dependent and independent variable
    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

# Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
y_pred_knn = model_knn.predict(X_test)
y_pred_svm = model_svm.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_gb = model_gb.predict(X_test)

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)

print("KNN Confusion matrix\n", conf_matrix_knn)
print("\nSVM Confusion matrix\n", conf_matrix_svm)
print("\nRF Confusion matrix\n", conf_matrix_rf)
print("\nGB Confusion matrix\n", conf_matrix_gb)

accuracy_knn = round(accuracy_score(y_test, y_pred_knn) * 100, 2)
accuracy_svm = round(accuracy_score(y_test, y_pred_svm) * 100, 2)
accuracy_rf = round(accuracy_score(y_test, y_pred_rf) * 100, 2)
accuracy_gb = round(accuracy_score(y_test, y_pred_gb) * 100, 2)

precision_knn = round(precision_score(y_test, y_pred_knn) * 100, 2)
precision_svm = round(precision_score(y_test, y_pred_svm) * 100, 2)
precision_rf = round(precision_score(y_test, y_pred_rf) * 100, 2)
precision_gb = round(precision_score(y_test, y_pred_gb) * 100, 2)

recall_knn = round(recall_score(y_test, y_pred_knn) * 100, 2)
recall_svm = round(recall_score(y_test, y_pred_svm) * 100, 2)
recall_rf = round(recall_score(y_test, y_pred_rf) * 100, 2)
recall_gb = round(recall_score(y_test, y_pred_gb) * 100, 2)

f1_knn = round(f1_score(y_test, y_pred_knn) * 100, 2)
f1_svm = round(f1_score(y_test, y_pred_svm) * 100, 2)
f1_rf = round(f1_score(y_test, y_pred_rf) * 100, 2)
f1_gb = round(f1_score(y_test, y_pred_gb) * 100, 2)

print("\n\nMetric\t\tKNN\t\tSVM\t\tRF\t\tGB")
print(f"Accuracy\t{accuracy_knn}\t{accuracy_svm}\t{accuracy_rf}\t{accuracy_gb}")
print(f"Precision\t{precision_knn}\t{precision_svm}\t{precision_rf}\t{precision_gb}")
print(f"Recall\t\t{recall_knn}\t{recall_svm}\t{recall_rf}\t{recall_gb}")
print(f"F1 Score\t{f1_knn}\t{f1_svm}\t{f1_rf}\t{f1_gb}")