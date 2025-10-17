def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without Normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


train_confusion_matrix = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(train_confusion_matrix, classes=np.unique(y_train), normalize=False, title="Confusion Matrix (Training Set)")
plt.show()

test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(test_confusion_matrix, classes=np.unique(y_test), normalize=False, title="Confusion Matrix (Test Set)")
plt.show()
