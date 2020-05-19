import numpy as np
import pandas as pd
import boto3
import warnings
warnings.filterwarnings("ignore")
from sklearn.externals import joblib
import os
import argparse
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from pandas import Series
import pmdarima as pm
from io import StringIO
from math import sqrt
#import currency_converter


import numpy as np

    
if __name__ == '__main__':
    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters

    # Data, model, and output directories
    parser.add_argument('--filename', type=str)
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
    data= pd.read_csv(os.path.join(args.train,args.filename))
    print(data)
    data['OrderReceivedDate'] = pd.to_datetime(data['OrderReceivedDate'])
    #test= pd.read_csv(os.path.join(args.test, args.test_file))
    data_usa=data[data.loc[:,"Country"]=="Usa"]

    #Convert Data to Volume based on Date
    data_usa.reset_index(drop=True, inplace=True)

    data_usa.loc[:,"Month"]=data_usa.loc[:,"OrderReceivedDate"].apply(lambda x : x.month)
    data_usa.loc[:,"Year"]=data_usa.loc[:,"OrderReceivedDate"].apply(lambda x : x.year)
    data_volume=data_usa.groupby(['Month','Year']).agg({"UniqueId":'count',"OrderReceivedDate":'first'}).reset_index().rename(columns={"UniqueId":"Volume"})
    print(data_volume)
    data_volume=data_volume[["Volume","OrderReceivedDate"]]
    data_volume=data_volume.sort_values(by='OrderReceivedDate')
    data_volume_train=data_volume.head(-12)
    data_volume.to_csv("./Volume.csv")
    #df= df.set_index('OrderReceivedDate')
    #df['OrderReceivedDate'] = pd.to_datetime(df['OrderReceivedDate'])
    data_volume = data_volume.set_index('OrderReceivedDate')

    print(data_volume)
    print(type(data_volume))
    data_volume.to_csv('./ActualVolume.csv')
    StringData = StringIO() 
    data_volume.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/ActualVolume.csv').put(Body=StringData.getvalue())

    data_volume_test=data_volume.tail(12)
    stepwise_fit = pm.auto_arima(data_volume_train.loc[:,"Volume"], start_p=1, start_q=1,
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal='M',
                             d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise

    print(stepwise_fit.summary())
    # pd.stepwise_fit.summary()
    x = stepwise_fit.summary()
    print(x)
    print(type(x))
    print(stepwise_fit.predict(n_periods=12))
    preds=stepwise_fit.predict(n_periods=12)
    mse = mean_squared_error(data_volume_test.loc[:,"Volume"],preds)
    original_test = data_volume_test.loc[:,"Volume"]
    preds_test=stepwise_fit.predict(n_periods=12)
    rmse_test = mean_squared_error(original_test,preds_test,squared =False)
    #mape_test = np.mean(abs(np.array(data_volume_test)-np.array(preds_test))/np.array(data_volume_test))*100 
    print(rmse_test)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)
    from io import StringIO
    lst=[]
    lst2=[]
    #lst3=[]
    #x=str(mape_test)
    #lst3.append(x)
    s=str(rmse_test)
    lst2.append(s)
    client = boto3.client('sagemaker',region_name='us-east-2')
    response = client.list_training_jobs(
        StatusEquals='InProgress',
        SortBy='Status',
    )
    print(response['TrainingJobSummaries'][0]['TrainingJobName'])
    Jobname=response['TrainingJobSummaries'][0]['TrainingJobName']
    lst.append(Jobname)
    df = pd.DataFrame(list(zip(lst, lst2)),columns =['Jobname', 'RMSE']) 
    StringData = StringIO() 
    df.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Accuracy/Accuracy-Arema.csv').put(Body=StringData.getvalue())
    joblib.dump(stepwise_fit, os.path.join(args.model_dir, "arima.joblib"))
    #y = joblib.load(os.path.join(args.model_dir, "arima.joblib"))
    #rmse_test = mean_squared_error(original_test,preds_test,squared =False)
    #mape_test = np.mean(abs(np.array(data_volume_test)-np.array(preds_test))/np.array(data_volume_test))*100 

    periods_train = data_volume_train.shape[0]
    preds_train = stepwise_fit.predict(n_periods= periods_train)
    actual_train = data_volume_train.loc[:,"Volume"]
    rmse_train = mean_squared_error(actual_train,preds_train,squared =False)
    mape_train = np.mean(abs(np.array(actual_train)-np.array(preds_train))/np.array(actual_train))*100 

    original_test = data_volume_test.loc[:,"Volume"]
    preds_test=stepwise_fit.predict(n_periods=12)
    rmse_test = mean_squared_error(original_test,preds_test,squared =False)
    mape_test = np.mean(abs(np.array(original_test)-np.array(preds_test))/np.array(original_test))*100 
    time_series_metrics = pd.DataFrame({"rmse_train":[round(rmse_train,2)],"rmse_test":[round(rmse_test,2)],"mape_train":[round(mape_train,2)],"mape_test":[round(mape_test,2)]})
    time_series_metrics.to_csv("time_series_metrics.csv",index=False)
    data_volume_train["fitted"] = preds_train
    data_volume_train.to_csv("time_Series_Actual_fitted.csv",index=False)
    StringData = StringIO() 
    time_series_metrics.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/time_series_metrics.csv').put(Body=StringData.getvalue())
    
    time_series_metrics.to_csv("time_series_metrics.csv",index=False)
    #print(data_volume_train.loc[:,"OrderReceivedDate"])
    #print(type(data_volume_train.loc[:,"OrderReceivedDate"]))

    data_volume_train["fitted"] = preds_train
    print(data_volume_train)
    StringData = StringIO() 
    data_volume_train.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/time_Series_Actual_fitted.csv').put(Body=StringData.getvalue())
    decomposition = sm.tsa.seasonal_decompose(data_volume.dropna(), model='additive',freq=12)
    decomposition.plot()
    seasonality = decomposition.seasonal ## write these three arrays to csv with corresponding headers
    trend = decomposition.trend
    residuals = decomposition.resid
    print(trend)
    print(type(trend))
    StringData = StringIO() 
    trend.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/trend.csv').put(Body=StringData.getvalue())
    trend.to_csv('./Trend.csv')
    StringData = StringIO() 
    seasonality.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/seasonality.csv').put(Body=StringData.getvalue())
    seasonality.to_csv('./seasonality.csv')
    StringData = StringIO() 
    residuals.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/residuals.csv').put(Body=StringData.getvalue())
    residuals.to_csv('./residuals.csv')
    #.predict(n_period=5)
    #print(y.predict(n_periods=int(test["month"])))

   
        
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        nums = []
        # Read the raw input data as CSV.
        df = pd.read_csv(io.StringIO(input_data), 
                         header=None)
        print(df)
        name=str(df)
        print(name)
        print(type(name))
        x=name.split(" ")
        print(x)
        while("" in x) : 
            x.remove("") 

        print(x[1]) 
        testdata=x[1]
        
        return testdata
    
    else:
        raise ValueError("{} not supported by script!")

def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().
    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    #print("##########################################")
    print(input_data)
    print(type(input_data))
    print("#########################################")
    #print(type(input_data[1]))
    print(input_data)
    features = model.predict(n_periods=int(input_data))
    print(features)
    return features

def model_fn(model_dir):
    """Deserialize fitted model
    """
    stepwise_fit = joblib.load(os.path.join(model_dir, "arima.joblib"))
    return stepwise_fit