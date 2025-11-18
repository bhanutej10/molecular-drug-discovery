Molecular Drug Discovery using Classical ML and Quantum VQC

This project explores the use of quantum machine learning for predicting small-molecule biological activity. It compares a classical Support Vector Machine (SVM) with a Variational Quantum Classifier (VQC) built using modern Qiskit primitives.

Objective

To evaluate whether quantum machine learning can support early-stage drug discovery by predicting whether a molecule is active or inactive, using simplified SMILES-based descriptors.

Method
Classical Model

Handcrafted SMILES descriptors

Min-Max scaling

PCA (4 components)

SVM with RBF kernel

Quantum Model (VQC)

Ry-based feature encoding

Two-layer ansatz with CZ entanglement

Statevector simulation

COBYLA optimization minimizing MSE loss

Uses only Qiskit 1.x compatible APIs

Repository Structure
molecular-drug-discovery/
├── notebooks/problem2_drug_vqc.ipynb
├── src/classical_model.py
├── src/quantum_vqc.py
├── src/utils.py
├── data/synthetic_smiles.csv
├── results/
├── README.md
└── requirements.txt

Results

The classical SVM provides a baseline.

The VQC converges successfully and learns a decision boundary.

Demonstrates an end-to-end quantum workflow for molecular classification.

Installation
pip install -r requirements.txt

Usage

Classical model:

python3 src/classical_model.py


Quantum VQC:

python3 src/quantum_vqc.py


Notebook:

notebooks/problem2_molecular.ipynb
