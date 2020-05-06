import boto3
import time
from botocore.vendored import requests
import pandas as pd
import io
import websocket
import pg

s3 = boto3.client('s3')
sm_client = boto3.client('sagemaker')
notebook_instance_name = 'bankingsa-aws-sagemaker-notebook'
raw_bucket = "sa_raw_zone"
published_bucket = "sa_published_zone"
bucket = "ptg-banking-poc"
file = "s3://sagemaker-us-east-1-754307369999/TrainData/DatasetEntry.xlsx"

url = sm_client.create_presigned_notebook_instance_url(
    NotebookInstanceName=notebook_instance_name)['AuthorizedUrl']


def connections():
    url_tokens = url.split('/')
    http_proto = url_tokens[0]
    http_hn = url_tokens[2].split('?')[0].split('#')[0]

    s = requests.Session()
    r = s.get(url)
    cookies = "; ".join(key + "=" + value for key, value in s.cookies.items())

    ws = websocket.create_connection(
        "wss://{}/terminals/websocket/1".format(http_hn),
        cookie=cookies,
        host=http_hn,
        origin=http_proto + "//" + http_hn
    )
    return ws


def get_last_modified(obj): return int(obj['LastModified'].strftime('%s'))


def read_file():
    db_endpoint = "sa2-db.cvl1on5n4pwi.us-east-1.rds.amazonaws.com"
    db_port = 5432
    db_name = "yash_sa_schema"
    db_username = "postgres"
    db_password = "sa2dbroot"
    conn = pg.DB(dbname=db_name, host=db_endpoint,
                 port=db_port, user=db_username, passwd=db_password)
    s3 = boto3.client('s3')
    objs = s3.list_objects_v2(Bucket='ptg-banking-poc',
                              Prefix='sa_published_zone/DS1/')['Contents']
    last_added = [obj['Key']
                  for obj in sorted(objs, key=get_last_modified, reverse=True)][0]
    file_upld_typ = 'Training'
    if(last_added.find('file_type=testing') > -1):
        file_upld_typ = 'Testing'
    if(last_added.find('file_type=retraining') > -1):
        file_upld_typ = 'Retraining'
    file_nm = last_added.split('/')[-1]
    q_result = conn.query(
        "SELECT file_upld_log_id FROM yash_sa_schema.sa_file_ingst_log sfil ORDER BY sfil.upld_ts DESC LIMIT 1")
    file_upld_log_id = q_result.dictresult()[0]['file_upld_log_id']
    last_added = 's3://ptg-banking-poc/'+last_added
    print(file_upld_log_id, last_added, file_nm)
    conn.query(
        "UPDATE yash_sa_schema.sa_file_ingst_log SET published_file_path='{}', published_file_name='{}' WHERE file_upld_log_id={};".format(last_added, file_nm, file_upld_log_id))
    conn.close()
    return file_upld_typ


def main():
    ws = connections()
    Type = read_file()
    print(Type)
    if Type == 'Training':
        ws.send(
            """[ "stdin", "jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/Generic.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1500\\r" ]""")
        time.sleep(1)
        ws.close()

    elif Type == 'Retraining':
        ws.send(
            """[ "stdin", "jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/GenericRetraining.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1500\\r" ]""")
        time.sleep(1)
        ws.close()

    elif Type == 'Testing':
        ws.send(
            """[ "stdin", "jupyter nbconvert --execute --to notebook --inplace /home/ec2-user/SageMaker/Testing.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1500\\r" ]""")
        time.sleep(1)
        ws.close()


if __name__ == '__main__':
    main()
