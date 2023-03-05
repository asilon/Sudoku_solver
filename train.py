import os
import joblib
import pickle
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def train_model():
    if os.path.exists('model/svm_model.pkl'):
        print("Loading existing model")
        return joblib.load('model/svm_model.pkl'), joblib.load('model/scaler.pkl')
    path = 'mnist.npz'
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path)

    n_samples, n_x, n_y = train_images.shape
    d2_train_dataset = train_images.reshape((n_samples, n_x * n_y))

    n_samples, n_x, n_y = test_images.shape
    d2_test_dataset = test_images.reshape((n_samples, n_x * n_y))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(d2_train_dataset)
    normalized_X_train = scaler.transform(d2_train_dataset)
    normalized_X_test = scaler.transform(d2_test_dataset)

    with open('model/scaler.pkl', 'wb') as out_file:
        pickle.dump(scaler, out_file)

    svc = SVC(kernel='rbf', gamma=0.0069, C=3.23)

    model = svc.fit(normalized_X_train, train_labels)

    train_predictions = svc.predict(normalized_X_train)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    print("train_accuracy: ", train_accuracy)

    pred = svc.predict(normalized_X_test)
    test_accuracy = accuracy_score(test_labels, pred)
    print("test accuracy: ", test_accuracy)

    # Save the model as a pickle in a file
    joblib.dump(svc, 'model/svc_model.pkl')
    return svc, scaler