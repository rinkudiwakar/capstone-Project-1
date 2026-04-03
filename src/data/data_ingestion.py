import numpy as np 
import pandas as pd 
import os
import logging
pd.set_option('future.no_silent_downcasting', True)
from src.logger.logging_file import logger
from src.connections import s3_connection
import yaml
from sklearn.model_selection import train_test_split

def load_data(data_url:str) ->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.info('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file : %s',e)
        raise
    except Exception as e : 
        logger.error ('Unexpected error occurred while loading the data: %s',e)
        raise


def load_params(params_path:str) -> dict:
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('arameter retrived from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('file not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error : %s',e)
        raise
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        logger.info('Preprocessing of data started.....')
        final_df= df[df['sentiment'].isin(['positive','negative'])]
        final_df['sentiment']=final_df['sentiment'].replace({
            'positive':1,
            'negative':0
        })
        logger.info('Preprocessing of the data is done....')
        return final_df
    except KeyError as e:
        logger.error('Missing column in the dataframe : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error during Preprocessing : %s',e)
        raise


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occured while savinf the data : %s",e)
        raise

def main():
    try:
        test_size=0.2
        #params=load_params(params_path='params.yaml')
        #test_size=params['data_ingestion']['test_size']
        
        df=load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        #s3=s3_connection.s3_operations('bucket-name','accesskey','secretkey')
        #df=s3.fetch_file_from_s3('data.csv')
        
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s',e)
        print(f"Error:{e}")

if __name__== '__main__':
    main()