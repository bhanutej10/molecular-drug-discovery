
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from utils import features_from_smiles

df = pd.read_csv('data/synthetic_smiles.csv')
X = np.array([features_from_smiles(s) for s in df.smiles])
y = df.active.values

sc = MinMaxScaler()
X = sc.fit_transform(X)

pca = PCA(n_components=4)
X = pca.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
y_train_s = 2*y_train - 1
y_test_s = 2*y_test - 1

n_qubits = 4
n_layers = 2
n_params = n_qubits * n_layers

def encoding(x):
    qc = QuantumCircuit(n_qubits)
    for i,v in enumerate(x):
        qc.ry(v*np.pi, i)
    return qc

def ansatz(params):
    qc = QuantumCircuit(n_qubits)
    idx=0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q); idx+=1
        for q in range(n_qubits-1):
            qc.cz(q, q+1)
    return qc

def build(x,params):
    qc = QuantumCircuit(n_qubits)
    qc.compose(encoding(x), inplace=True)
    qc.compose(ansatz(params), inplace=True)
    return qc

def exp_z0(statevector):
    probs = np.abs(statevector.data)**2
    exp=0
    for idx,p in enumerate(probs):
        bit0 = idx & 1
        exp += (1 if bit0==0 else -1)*p
    return exp

def predict_ev(x,params):
    sv = Statevector.from_instruction(build(x,params))
    return exp_z0(sv)

def loss(params):
    preds = np.array([predict_ev(x,params) for x in X_train])
    return np.mean((preds - y_train_s)**2)

init = 0.1*np.random.randn(n_params)
res = minimize(loss, init, method='COBYLA', options={'maxiter':200})
opt = res.x

preds = np.array([1 if predict_ev(x,opt)>0 else 0 for x in X_test])
print("VQC Accuracy:", accuracy_score(y_test,preds))
print(classification_report(y_test,preds))
