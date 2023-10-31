import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from sklearn import metrics

####defining training path
train_data_path="fashion-mnist_train.csv"

####loading train data
train_data= pd.read_csv(train_data_path)

####defining dependent and independent variable

X=train_data.drop("label",axis=1)
y=train_data['label']

########visualizing tha data for checking the imbalance 
pot_lbl = y.value_counts()
# Barplot
plt.figure(figsize=(8,5))
sns.barplot(x=pot_lbl.index, y=pot_lbl)
plt.xlabel('result', fontsize=15)
plt.ylabel('count', fontsize=15)
# Display the plot
plt.savefig('Imbalance_checking.png')
plt.show()



#######normalize the data to range(0,1)
X=X/255.0


def train_and_save_model(model_type,X,y):
    if model_type == 'ML':
        model_1 = xgb.XGBClassifier()
        model_2 = HistGradientBoostingClassifier()
        final_model = VotingClassifier(estimators=[('xgb', model_1), ('HGB', model_2)], voting='soft')
        final_model.fit(X, y)
        model_name = 'ensemble_ml.pkl'
    elif model_type == 'DL':


        X=np.array(X)
        X = X.reshape(-1, 28, 28, 1)
        height,width = 28,28
        channels=1
        num_classes=10
        # Define the model
        final_model = Sequential()

        # Add convolutional layers
        final_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
        final_model.add(MaxPooling2D((2, 2)))
        final_model.add(Conv2D(64, (3, 3), activation='relu'))
        final_model.add(MaxPooling2D((2, 2)))
        final_model.add(Conv2D(128, (3, 3), activation='relu'))
        final_model.add(MaxPooling2D((2, 2)))

        # Flatten the output of the convolutional layers
        final_model.add(Flatten())

        # Add a dense layer for classification
        final_model.add(Dense(128, activation='relu'))
        final_model.add(Dense(64, activation='relu'))
        final_model.add(Dense(32, activation='relu'))
        final_model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # print(final_model.summary())

        # Train the model using your dataset
        num_epochs=20
        batch_size=32
        history=final_model.fit(X,y, epochs=num_epochs, batch_size=batch_size)

        model_name = 'ensemble_dl.pkl'
    else:
        print("Invalid model type. Please choose 'ML' or 'DL'.")
        return
    
    joblib.dump(final_model, model_name)
    print(f"Model training done. Saved as {model_name}")
    print("Run the evaluate_model.py file next.")

# Get user input to choose ML or DL model
model_type = input("Choose model type (ML or DL): ").strip().upper()
train_and_save_model(model_type,X,y)