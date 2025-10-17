acc_train_list = {
'cnn': mlp_train_accuracy,
'stack': stack_model_train_accuracy}

mcc_train_list = {
'svm_rbf': train_mcc,
'mlp': mlp_train_mcc,
'stack': stack_model_train_mcc}

f1_train_list = {
'svm_rbf': train_f1_micro,
'mlp': mlp_train_f1,
'stack': stack_model_train_f1}

acc_train_list
mcc_train_list
f1_train_list
import pandas as pd

acc_df = pd.DataFrame.from_dict(acc_train_list, orient='index', columns=['Accuracy'])
mcc_df = pd.DataFrame.from_dict(mcc_train_list, orient='index', columns=['MCC'])
f1_df = pd.DataFrame.from_dict(f1_train_list, orient='index', columns=['F1'])
df = pd.concat([acc_df, mcc_df, f1_df], axis=1)

df.to_csv('results.csv')
