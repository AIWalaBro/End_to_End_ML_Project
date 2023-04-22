import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from  dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()
        
    def initiated_data_ingestion(self):
        logging.info('enter into the data ingestion components')
        try:
            df = pd.read_csv(r'D:\ML_Projects\notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')
            
            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path), exist_ok = True)
            
            
            
            # read the dataset
            df.to_csv(self.Ingestion_config.raw_data_path, index = False, header = True)
            
            
            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
    
            train_set.to_csv(self.Ingestion_config.train_data_path,index = False, header =True )
            
            test_set.to_csv(self.Ingestion_config.test_data_path,index = False, header =True )
            
            
            logging.info("ingestion of the data has been completed")
            
            return (
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
                
                    )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiated_data_ingestion()
        
    
    
