import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO
# from mode_train import model_type


parser = argparse.ArgumentParser(description='Model Evaluation')

# Define the command-line argument for the test data path
parser.add_argument('--test-data', default='fashion-mnist_test.csv', help='Path to the test data file')
# Parse the command-line arguments
args = parser.parse_args()

test_path=args.test_data

####loading test data
test_data= pd.read_csv(test_path)

####defining dependent and independent variable

X_test=test_data.drop("label",axis=1)
y_test=test_data['label']
 
X_test=X_test/255.0

##########loading the model
# Get user input to choose ML or DL model
model_type = input("Choose model type (ML or DL): ").strip().upper()
if model_type=="ML":
    final_model =  joblib.load('ensemble_ml.pkl')
    pred = final_model.predict(X_test)

    testing=accuracy_score(y_test, pred)
    print("testing acc: ",testing)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, pred))

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix.png')
    plt.show()
    # Save the figure to a file


    classification_rep = classification_report(y_test, pred)
    # Save the classification report to a text file
    with open('output.txt', 'w') as file:
        file.write(classification_rep)
else:
    X_test=np.array(X_test)
    X_test = X_test.reshape(-1, 28, 28, 1)
    final_model =  joblib.load('ensemble_dl.pkl')
    print(final_model.summary())
    # Evaluate the model
    test_loss, test_accuracy = final_model.evaluate(X_test, y_test)

    # Make predictions
    predictions = final_model.predict(X_test)
    predictions=np.argmax(predictions,axis=1)


    testing=accuracy_score(y_test, predictions)
    print("testing acc: ",testing)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig('confusion_matrix.png')
    plt.show()
    # Save the figure to a file


    classification_rep = classification_report(y_test, predictions)
    model_summary=final_model.summary()
 
    sio = StringIO()
    final_model.summary(print_fn=lambda x: sio.write(x + '\n'))
    model_summary = sio.getvalue()
    sio.close()
    # Save the classification report to a text file
    with open('output.txt', 'w') as file:
        file.write(classification_rep)
        file.write('\n\n')
        file.write(model_summary)







