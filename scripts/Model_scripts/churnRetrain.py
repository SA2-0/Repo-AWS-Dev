# Importing the libraries
import numpy as np
import pandas as pd
import boto3

import warnings
warnings.filterwarnings("ignore")
from sklearn.externals import joblib
import os
import argparse
from sklearn.metrics import mean_squared_error
import io
from math import sqrt
import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters

    # Data, model, and output directories
    parser.add_argument('--filename', type=str)
    parser.add_argument('--batchsize', type=str, default='10')
    parser.add_argument('--epochs', type=str, default='20')
    #parser.add_argument('--n_periods', type=str, default='{"n_periods": 5}')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    #parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='cleansed_data.xlsx')
    #parser.add_argument('--test-file', type=str, default='arimatest.csv')
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features
    parser.add_argument('--target', type=str)  # in this script we ask user to explicitly name the target
    #parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    #parser.add_argument('--test-file', type=str, default='testdata1.csv')

    args, _ = parser.parse_known_args()
    print(args.filename)
    print(args.train)
    print(args.batchsize)
    dataset= pd.read_csv(os.path.join(args.train,args.filename))
    print(dataset)

    # Importing the dataset
    #dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:13].values
    y = dataset.iloc[:, 13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    labelencoder_X_2 = LabelEncoder()

    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    #onehotencoder = OneHotEncoder(categorical_features = [1])

    #X = onehotencoder.fit_transform(X).toarray()

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

    X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

    X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    batch=int(args.batchsize)
    epoch=int(args.epochs)
    print(args.epochs)
    print(type(args.batchsize))
    print(type(args.epochs))
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = batch, epochs = epoch)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    from sklearn.metrics import confusion_matrix
    accuracy = confusion_matrix(y_test, y_pred).ravel()
    #accuracy=np.mean(abs(np.array(y_test)-np.array(y_pred))/np.array(y_pred))*100 
    print(accuracy)
    s=str(accuracy)
    print(s[1:-1])
    y=s[1:-1]
    print(y)
    print(type(y))
    x=y.split(" ")
    print(x)
    Newlr=[z for z in x if z]
    print(type(Newlr))
    Numerator=int(Newlr[0])+int(Newlr[3])
    Denominator=int(Newlr[0])+int(Newlr[1])+int(Newlr[2])+int(Newlr[3])
    print(Numerator)
    print(Denominator)
    Accuracy=Numerator/Denominator
    print(Accuracy*100)
    accuracy=Accuracy*100
    from io import StringIO
    lst=[]
    lst2=[]
    s=str(accuracy)
    lst2.append(s)
    client = boto3.client('sagemaker',region_name='us-east-2')
    response = client.list_training_jobs(
        StatusEquals='InProgress',
        SortBy='Status',
    )
    print(response['TrainingJobSummaries'][0]['TrainingJobName'])
    Jobname=response['TrainingJobSummaries'][0]['TrainingJobName']
    lst.append(Jobname)
    df = pd.DataFrame(list(zip(lst, lst2)),columns =['Jobname', 'MAPE']) 
    StringData = StringIO() 
    df.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Accuracy/Accuracy-ChurnRetrain'+str(accuracy)+'.csv').put(Body=StringData.getvalue())

    joblib.dump(classifier, os.path.join(args.model_dir, "Churn.joblib"))


    

    
    
    

def input_fn(input_data, content_type):
    if content_type == 'text/csv':
        nums = []
            # Read the raw input data as CSV.
        dataset = pd.read_csv(io.StringIO(input_data), 
                             header=None)
       
        return dataset

    else:
        raise ValueError("{} not supported by script!")
        
def model_fn(model_dir):
    classifier = joblib.load(os.path.join(model_dir, "Churn.joblib"))
    return classifier

def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    #print("##########################################")
    #print(input_data)
    #print(type(input_data))
    #print("#########################################")
    #print(type(input_data[1]))
    #print(input_data)
    X = input_data.iloc[:, 3:13].values
    y = input_data.iloc[:, 13].values

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    labelencoder_X_2 = LabelEncoder()

    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

    #onehotencoder = OneHotEncoder(categorical_features = [1])

    #X = onehotencoder.fit_transform(X).toarray()

    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')

    X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

    X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    features = model.predict(X_test)
    y_pred = (features > 0.5)
    print("#######################################################")
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("cm value is=")
    print(cm)
    print(type(cm))
    print("##########################################")
    return y_pred
