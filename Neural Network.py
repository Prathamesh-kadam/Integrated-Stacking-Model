import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

n_samples_train = X_train.shape[0]
n_features_train = np.prod(X_train.shape[1:])
X_train_2d = X_train.reshape(n_samples_train, n_features_train)

n_samples_test = X_test.shape[0]
n_features_test = np.prod(X_test.shape[1:])
X_test_2d = X_test.reshape(n_samples_test, n_features_test)

y_train_flat = y_train.ravel()
y_test_flat = y_test.ravel()

mlp = MLPClassifier(alpha=1, max_iter=1000, random_state=42)  
mlp.fit(X_train_2d, y_train_flat)

y_train_pred = mlp.predict(X_train_2d)
y_test_pred = mlp.predict(X_test_2d)

mlp_train_accuracy = accuracy_score(y_train_flat, y_train_pred)   
mlp_train_mcc = matthews_corrcoef(y_train_flat, y_train_pred) 
mlp_train_f1 = f1_score(y_train_flat, y_train_pred, average='weighted') 

mlp_test_accuracy = accuracy_score(y_test_flat, y_test_pred)  
mlp_test_mcc = matthews_corrcoef(y_test_flat, y_test_pred)
mlp_test_f1 = f1_score(y_test_flat, y_test_pred, average='weighted')  

print('Model performance for Training set')
print('- Accuracy:', mlp_train_accuracy)
print('- MCC:', mlp_train_mcc)
print('- F1 score:', mlp_train_f1)
print('----------------------------------')
print('Model performance for Test set')
print('- Accuracy:', mlp_test_accuracy)
print('- MCC:', mlp_test_mcc)
print('- F1 score:', mlp_test_f1)
