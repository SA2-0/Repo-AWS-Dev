import pandas as pd 
import numpy as np 
import warnings 
from sklearn.externals import joblib 
import os 
import argparse
import io
import boto3
import numpy as np 
#import currency_converter 
from currency_converter import CurrencyConverter 
import quandl
from sklearn.metrics import r2_score
from io import StringIO
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
import statsmodels.api as sm

 
def convert_currency(x, typ): 
    c=CurrencyConverter() 
    return c.convert(x,typ, "USD") 

if __name__ == '__main__': 
    print('extracting arguments') 
    parser = argparse.ArgumentParser() 
 
    # hyperparameters sent by the client are passed as command-line arguments to the script. 
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters 
 
    parser.add_argument('--filename', type=str) 
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN')) 
    #parser.add_argument('--train-file', type=str, default='CleansedData.xlsx') 
    parser.add_argument('--features', type=str)  # in this script we ask user to explicitly name features 
    parser.add_argument('--target', type=str)  # in this script we ask user to explicitly name the target 
    #parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST')) 
    #parser.add_argument('--test-file', type=str, default='testdata1.csv') 
 
    args, _ = parser.parse_known_args()
    data_cleansed= pd.read_csv(os.path.join(args.train, args.filename))
    data_cleansed['OrderReceivedDate'] = pd.to_datetime(data_cleansed['OrderReceivedDate'])
    corr=data_cleansed.corr()
    corr.to_csv('./CorrelationMatrixLinear.csv')
    StringData = StringIO() 
    corr.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/Correlation_metrics.csv').put(Body=StringData.getvalue())
    #s3_resource.Object(bucket_name='sagemaker-us-east-1-754307369999',key='Output/Correlation_metrics.csv').put(Body=StringData.getvalue())

    
    #data_cleansed=pd.read_excel( "D:    My_Projects    Hyster-Yale    Python    Cleansed Data.xlsx ", encoding='ISO-8859-1') 
 
    data_usa=data_cleansed[data_cleansed.loc[:,"Country"]=="Usa"]
    data_usa.loc[:,"Adj_OrderGrossValue"]=data_usa.loc[:,"ordergrossvalue"]
    data_usa.loc[:,"Adj_OrderNetValue"]=data_usa.loc[:,"ordernetvalue"]
    data_cad=data_usa[data_usa.loc[:,"currencycode"]=="CAD"]
    data_cad.loc[:,"Adj_OrderGrossValue"]=data_cad.loc[:,"ordergrossvalue"].apply(lambda x: convert_currency(x,"CAD"))
    data_cad.loc[:,"Adj_OrderNetValue"]=data_cad.loc[:,"ordernetvalue"].apply(lambda x: convert_currency(x,"CAD"))

    data_jpy=data_usa[data_usa.loc[:,"currencycode"]=="EUR"]
    data_jpy.loc[:,"Adj_OrderGrossValue"]=data_jpy.loc[:,"ordergrossvalue"].apply(lambda x: convert_currency(x,"JPY"))
    data_jpy.loc[:,"Adj_OrderNetValue"]=data_jpy.loc[:,"ordernetvalue"].apply(lambda x: convert_currency(x,"JPY"))

    data_usd=data_usa[data_usa.loc[:,"currencycode"]=="USD"]

    import quandl
    quandl.ApiConfig.api_key = 'qyJmR3Jiz253zhrGS92n'
    #Consumer Price Index
    df_cpi=pd.DataFrame(quandl.get('FRED/CPIAUCSL', column_index='1'))
    df_cpi.loc[:,"Date"]=df_cpi.index
    df_cpi.reset_index(drop=True, inplace=True)
    df_cpi.loc[:,"Month"]=df_cpi.loc[:,"Date"].apply(lambda x : x.month)
    df_cpi.loc[:,"Year"]=df_cpi.loc[:,"Date"].apply(lambda x : x.year)
    df_cpi=df_cpi.groupby(['Month','Year']).agg({"Value":'count',"Date":'first'}).reset_index().rename(columns={"Value":"CPI"})
    df_cpi.loc[:,"DateFinal"]=pd.to_datetime(df_cpi['Date']).dt.to_period('M')
    df_cpi=df_cpi[["DateFinal","CPI"]]


    #Industrial Production Index
    df_inpi=pd.DataFrame(quandl.get('FRED/INDPRO', column_index='1'))
    df_inpi.loc[:,"Date"]=df_inpi.index
    df_inpi.reset_index(drop=True, inplace=True)
    df_inpi.loc[:,"Month"]=df_inpi.loc[:,"Date"].apply(lambda x : x.month)
    df_inpi.loc[:,"Year"]=df_inpi.loc[:,"Date"].apply(lambda x : x.year)
    df_inpi=df_inpi.groupby(['Month','Year']).agg({"Value":'count',"Date":'first'}).reset_index().rename(columns={"Value":"INPI"})
    df_inpi.loc[:,"DateFinal"]=pd.to_datetime(df_inpi['Date']).dt.to_period('M')
    df_inpi=df_inpi[["DateFinal","INPI"]]


    #Gross Private Domestic Investment
    df_gpdi=pd.DataFrame(quandl.get('FRED/GPDI', column_index='1'))
    df_gpdi.loc[:,"Date"]=df_gpdi.index
    df_gpdi.reset_index(drop=True, inplace=True)
    df_gpdi.loc[:,"Month"]=df_gpdi.loc[:,"Date"].apply(lambda x : x.month)
    df_gpdi.loc[:,"Year"]=df_gpdi.loc[:,"Date"].apply(lambda x : x.year)
    df_gpdi=df_gpdi.groupby(['Month','Year']).agg({"Value":'count',"Date":'first'}).reset_index().rename(columns={"Value":"GPDI"})
    df_gpdi.loc[:,"DateFinal"]=pd.to_datetime(df_gpdi['Date']).dt.to_period('M')
    df_gpdi=df_gpdi[["DateFinal","GPDI"]]

    #Total Public Debt
    df_pubdeb=pd.DataFrame(quandl.get('FRED/GFDEBTN', column_index='1'))
    df_pubdeb.loc[:,"Date"]=df_pubdeb.index
    df_pubdeb.reset_index(drop=True, inplace=True)
    df_pubdeb.loc[:,"Month"]=df_pubdeb.loc[:,"Date"].apply(lambda x : x.month)
    df_pubdeb.loc[:,"Year"]=df_pubdeb.loc[:,"Date"].apply(lambda x : x.year)
    df_pubdeb=df_pubdeb.groupby(['Month','Year']).agg({"Value":'count',"Date":'first'}).reset_index().rename(columns={"Value":"PUBDEB"})
    df_pubdeb.loc[:,"DateFinal"]=pd.to_datetime(df_pubdeb['Date']).dt.to_period('M')
    df_pubdeb=df_pubdeb[["DateFinal","PUBDEB"]]

    #Total Commercial and Industrial Loans
    df_totci=pd.DataFrame(quandl.get('FRED/TOTCI', column_index='1'))
    df_totci.loc[:,"Date"]=df_totci.index
    df_totci.reset_index(drop=True, inplace=True)
    df_totci.loc[:,"Month"]=df_totci.loc[:,"Date"].apply(lambda x : x.month)
    df_totci.loc[:,"Year"]=df_totci.loc[:,"Date"].apply(lambda x : x.year)
    df_totci=df_totci.groupby(['Month','Year']).agg({"Value":'count',"Date":'first'}).reset_index().rename(columns={"Value":"TOTCI"})
    df_totci.loc[:,"DateFinal"]=pd.to_datetime(df_totci['Date']).dt.to_period('M')

    df_totci=df_totci[["DateFinal","TOTCI"]]


    data_combined=pd.concat([data_usd, data_cad, data_jpy])
    data_combined.loc[:,"Month"]=data_combined.loc[:,"OrderReceivedDate"].apply(lambda x : x.month)
    data_combined.loc[:,"Year"]=data_combined.loc[:,"OrderReceivedDate"].apply(lambda x : x.year)
    data_volume=data_combined.groupby(['Month','Year']).agg({"UniqueId":'count',"OrderReceivedDate":'first'}).reset_index().rename(columns={"UniqueId":"Volume"})
    data_volume.loc[:,"DateFinal"]=pd.to_datetime(data_volume['OrderReceivedDate']).dt.to_period('M')
    data_volume=data_volume[["Volume","DateFinal"]]

    df_macro_econ_fac_1=pd.merge(df_cpi, df_inpi, on='DateFinal', how='outer')
    df_macro_econ_fac_2=pd.merge(df_macro_econ_fac_1, df_gpdi, on='DateFinal', how='outer')
    df_macro_econ_fac_3=pd.merge(df_macro_econ_fac_2, df_pubdeb, on='DateFinal', how='outer')
    df_macro_econ_fac_4=pd.merge(df_macro_econ_fac_3, df_totci, on='DateFinal', how='outer')

    data_final=data_volume.merge(df_macro_econ_fac_4, left_on='DateFinal',right_on='DateFinal', how='left')

    data_final = data_final.sort_values(by='DateFinal',ascending=True)

    data_vol=data_final.loc[:,"Volume"]
    lagged_vols=pd.concat([data_vol.shift(1), data_vol.shift(2), data_vol.shift(3)], axis=1)
    lagged_vols.columns=["Vol_Lag_1", "Vol_Lag_2", "Vol_Lag_3"]

    df_final=pd.concat([data_final, lagged_vols], axis=1)
    df_final=df_final[3:]

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    df_final.loc[:,"Vol_Lag_1"]=df_final.loc[:,"Vol_Lag_1"].apply(lambda x: np.log(x+1))
    df_final.loc[:,"Vol_Lag_2"]=df_final.loc[:,"Vol_Lag_2"].apply(lambda x: np.log(x+1))
    df_final.loc[:,"Vol_Lag_3"]=df_final.loc[:,"Vol_Lag_3"].apply(lambda x: np.log(x+1))
    NewTest_Df=df_final.loc[:,['Volume', 'DateFinal']]
    print(NewTest_Df)
    StringData = StringIO() 
    NewTest_Df.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/Linear_Actual.csv').put(Body=StringData.getvalue())
    #Correlation_matrix changes
    data_discount = data_usa.loc[:,["OrderReceivedDate","Discount","ordernetvalue"]]
    data_discount.loc[:,"month"]=data_discount.loc[:,"OrderReceivedDate"].apply(lambda x : x.month)
    data_discount.loc[:,"year"]=data_discount.loc[:,"OrderReceivedDate"].apply(lambda x : x.year)
    data_discount = data_discount.groupby(["year","month"]).agg({"Discount":'mean',"ordernetvalue":'mean',"OrderReceivedDate":'first'})
    data_discount["Discount"]=round(data_discount["Discount"],2)
    data_discount["ordernetvalue"]=round(data_discount["ordernetvalue"],2)
    data_discount = data_discount.reset_index()
    data_discount.loc[:,"DateFinal"]=pd.to_datetime(data_discount['OrderReceivedDate']).dt.to_period('M')
    #print(data_discount.head(10))
    data_discount = data_discount.loc[:,["Discount","ordernetvalue","DateFinal"]]
    #print(data_discount.head())
    data_discount.columns = ["Avg_Discount","Avg_order_value","DateFinal"]
    #print(data_discount.head(10))
    #print(data_final.head(10))
    data_all_features = df_final.merge(data_discount,on='DateFinal')

    df_volume = data_final.loc[:,["DateFinal","Volume"]]
    df_volume = df_volume.set_index('DateFinal')
    decomposition = sm.tsa.seasonal_decompose(df_volume.dropna(), model='additive',period=12)
    seasonality = decomposition.seasonal
    print(seasonality)
    trend = decomposition.trend
    trend.columns = ["trend"]
    seasonality.columns = ["seasonality"]
    data_all_features = data_all_features.merge(trend,on='DateFinal',how='left')
    data_all_features = data_all_features.merge(seasonality,on='DateFinal',how='left')## FINAL DATAFRAME with all imp variables 
    print(data_all_features)
    data_all_features_corr = data_all_features.loc[:,['Volume', 'Vol_Lag_1', 'Vol_Lag_2','Vol_Lag_3','trend', 'seasonality',"Avg_Discount","Avg_order_value"]]
    corr_matrix_final = data_all_features_corr.corr()
    print(corr_matrix_final)
    corr_matrix_final.to_csv("new_corr_matrix.csv")
    StringData = StringIO() 
    corr_matrix_final.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/Correlation_matrix_lr.csv').put(Body=StringData.getvalue())
    
    
    #--------------------------------------------
    df_final_cols_removed=df_final.loc[:,['Volume', 'DateFinal', 'CPI', 'INPI', 'TOTCI', 'Vol_Lag_1', 'Vol_Lag_2', 'Vol_Lag_3']]
    X=df_final_cols_removed.loc[:,['CPI', 'INPI', 'TOTCI', 'Vol_Lag_1', 'Vol_Lag_2', 'Vol_Lag_3']]
    y=df_final_cols_removed.loc[:,"Volume"]
    import statsmodels.api as sm
    mod = sm.OLS(y,X)
    fii = mod.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    X_train=X.head(-12)
    y_train=y.head(-12)
    X_test=X[-12:-3]
    y_test=y[-12:-3]
    lr=LinearRegression().fit(X_train, y_train)
    coeff=lr.coef_
    y_pred=lr.predict(X_test)
    print(X_train)# predictions on test data. SAVE this to one csv for model graph
    y_train_pred = lr.predict(X_train) # predictions on train data. SAVE this to one csv for model graph
    print(y_train_pred)
    s = str(y_train_pred)
    #joblib.dump(lr, os.path.join(args.model_dir,"lr.joblib"))

    rfr=RandomForestRegressor(n_estimators=1000, criterion='mae').fit(X_train, y_train)
    #rfr.fit(X_train, y_train)
    y_pred_rfr=rfr.predict(X_test)
    joblib.dump(rfr, os.path.join(args.model_dir,"rfr.joblib"))
    
    #s = y_train_pred.decode()
    #print(s)
    print(s[1:-1])
    y=s[1:-1]
    x=y.split(" ")
    print(x)
    print([s.replace('\n', '') for s in x])
    without_empty_strings = []
    for string in x:
        if (string != '\n'):
            without_empty_strings.append(string)


    print(without_empty_strings)
    df1 = pd.DataFrame(without_empty_strings)
    lr = []
    lr.append(x)

    #print(lr)
    df = pd.DataFrame(lr)
    newDF=df.T
    newDF=newDF.dropna()
    StringData = StringIO() 
    newDF.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/LinearTrainOutput.csv').put(Body=StringData.getvalue())


    accuracy=np.mean(abs(np.array(y_test)-np.array(y_pred))/np.array(y_test))*100  # this is mape on test data.SAVE
    print(accuracy)
    mape_train = np.mean(abs(np.array(y_train)-np.array(y_train_pred))/np.array(y_train))*100  # mape on train data.SAVE
    r_squared = r2_score(y_test,y_pred)# to display rsquared value on right SAVE
    print(r_squared)
    print(type(r_squared))

    metrics = {'Train_mape':mape_train , 'Test_mape':accuracy, 'r_squared':r_squared}
    df_s = pd.DataFrame(metrics, index=[0])
    StringData = StringIO() 
    df_s.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/metrics.csv').put(Body=StringData.getvalue())
     
    #final_lr=LinearRegression() 
    #final_lr.fit(X[:-3], y[:-3]) 
    accuracy=np.mean(abs(np.array(y_test)-np.array(y_pred_rfr))/np.array(y_pred_rfr))*100 
    print(accuracy)
    from io import StringIO
    lst=[]
    lst2=[]
    s=str(accuracy)
    lst2.append(s)
    client = boto3.client('sagemaker',region_name='us-east-1')
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
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Accuracy/Accuracy-Linearretrain'+str(accuracy)+'.csv').put(Body=StringData.getvalue())
    coeff_df = pd.DataFrame(coeff, X.columns, columns=['coefficients']) #draw coefficient table SAVE
    
    pred_model_coeff = coeff_df.merge(p_values, how= 'inner', right_index = True, left_index = True)
    pred_model_coeff = pred_model_coeff.reset_index()
    pred_model_coeff .columns = ['features','coefficients','p_value']
    print(pred_model_coeff)
    pred_model_coeff.to_csv("prediction_model_coefficients.csv",index=False)
    StringData = StringIO() 
    pred_model_coeff.to_csv(StringData)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name='sa2-train-bucket',key='Output/prediction_model_coefficients.csv').put(Body=StringData.getvalue())
     



def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    import io
    if content_type == 'text/csv':
        nums = []
        # Read the raw input data as CSV.
        df = pd.read_csv(io.StringIO(input_data), 
                         header=None)
        print(df)

        return df

    else:
        raise ValueError("{} not supported by script!")

def predict_fn(input_data, model):
    print(input_data)
    print(type(input_data))
    print("#########################################")
    #print(type(input_data[1]))
    print(input_data)
    features = model.predict(input_data)
    print(features)
    return features

def model_fn(model_dir):
    rfr = joblib.load(os.path.join(model_dir, "rfr.joblib"))
    return rfr