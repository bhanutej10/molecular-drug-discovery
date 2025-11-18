
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def features_from_smiles(s):
    return [
        len(s),
        sum(1 for c in s if c.isupper()),
        sum(1 for c in s if c.isdigit()),
        s.count('C')
    ]

df = pd.read_csv('data/synthetic_smiles.csv')
X = np.array([features_from_smiles(s) for s in df.smiles])
y = df.active.values

sc = MinMaxScaler()
X = sc.fit_transform(X)

pca = PCA(n_components=4)
X = pca.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

svm = SVC(kernel='rbf')
svm.fit(X_train,y_train)
pred = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test,pred))
print(classification_report(y_test,pred))
