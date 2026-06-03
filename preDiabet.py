import requests
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    average_precision_score
)

def main():
    downloader()
    model, X_train, y_train, X_valid, y_valid = preprocess()
    if model is not None: 
        history_df = train_model(model, X_train, y_train, X_valid, y_valid)
        visualize(history_df)


def downloader(filename='diabet.csv'):
    url = 'https://drive.google.com/uc?id=1r7avcqz1wm7_2NYb_gCraCiPYWZC1AxK'
    response = requests.get(url)
    response.raise_for_status() 

    diabetfile = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    file_path = os.path.join(filename)
    diabetfile.to_csv(file_path, index=False)

def preprocess():
    diabet = pd.read_csv("diabet.csv")
    print(diabet.head()) 

    X = diabet.copy()
    y = X.pop('Outcome')

    X.set_index('Id', inplace=True)
    print(X.head())  

    features_num = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    transformer_num = make_pipeline(
        SimpleImputer(strategy="constant"), 
        StandardScaler(),
    )
    preprocessor = make_column_transformer(
        (transformer_num, features_num)
    )

    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, stratify=y, train_size=0.75)

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    input_shape = X_train.shape[1]
    
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model, X_train, y_train, X_valid, y_valid
def train_model(model, X_train, y_train, X_valid, y_valid):
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.00001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=64,
        epochs=200,
        callbacks=[early_stopping],
    )
    history_df = pd.DataFrame(history.history)
    return history_df

def visualize(history_df):
    # Plot loss
    plt.figure(figsize=(12, 5))
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
    plt.xlabel("Epochs")  
    plt.ylabel("Loss")    
    plt.legend(["Training Loss", "Validation Loss"])
    plt.grid()  
    plt.savefig('loss_plot.png') 
    plt.close()  

    # Plot accuracy
    plt.figure(figsize=(12, 5))  
    history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
    plt.xlabel("Epochs")  
    plt.ylabel("Accuracy")  
    plt.legend(["Training Accuracy", "Validation Accuracy"])  
    plt.grid()  
    plt.savefig('accuracy_plot.png') 
    plt.close()  

if __name__ == "__main__":
    main()