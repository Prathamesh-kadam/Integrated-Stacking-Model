import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import matthews_corrcoef, f1_score

n_samples_train = X_train.shape[0]
n_features_train = np.prod(X_train.shape[1:])
X_train_2d = X_train.reshape(n_samples_train, n_features_train)

n_samples_test = X_test.shape[0]
n_features_test = np.prod(X_test.shape[1:])
X_test_2d = X_test.reshape(n_samples_test, n_features_test)

y_train_flat = y_train[:n_samples_train].ravel()
y_test_flat = y_test[:n_samples_test].ravel()

class SVMClassifier(SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = np.prod(X.shape[1:])
        X_2d = X.reshape(n_samples, n_features)
        y_flat = y.ravel()
        return super().fit(X_2d, y_flat)

    def predict(self, X):
        n_samples = X.shape[0]
        n_features = np.prod(X.shape[1:])
        X_2d = X.reshape(n_samples, n_features)
        return super().predict(X_2d)

svm_rbf = SVMClassifier(kernel='rbf', gamma=2, C=1)

svm_rbf.fit(X_train_2d, y_train_flat)

estimator_list = [
    ('svm', SVMClassifier(gamma='scale'))
]

voting_classifier = VotingClassifier(estimators=estimator_list, voting='hard')

voting_classifier.fit(X_train_2d, y_train_flat)

y_train_pred = voting_classifier.predict(X_train_2d)
y_test_pred = voting_classifier.predict(X_test_2d)

train_mcc = matthews_corrcoef(y_train_flat, y_train_pred)
train_f1_macro = f1_score(y_train_flat, y_train_pred, average='macro')
train_f1_micro = f1_score(y_train_flat, y_train_pred, average='micro')

test_mcc = matthews_corrcoef(y_test_flat, y_test_pred)
test_f1_macro = f1_score(y_test_flat, y_test_pred, average='macro')
test_f1_micro = f1_score(y_test_flat, y_test_pred, average='micro')

print('Training set:')
print('MCC:', train_mcc)
print('F1-score (macro):', train_f1_macro)
print('F1-score (micro):', train_f1_micro)
print('------------------------')
print('Test set:')
print('MCC:', test_mcc)
print('F1-score (macro):', test_f1_macro)
print('F1-score (micro):', test_f1_micro)
