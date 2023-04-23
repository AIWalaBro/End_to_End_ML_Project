import os
import sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from  dataclasses import dataclass

# extractoing from data_transformation folder to check whether our taindata and testdata
# sucessfully transform or not.

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationconfig

# checking our best model
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModeltrainerConfig

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
            
            
            logging.info("Ingestion of the data has been completed")
            
            return (
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
                
                    )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    
    # combining data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiated_data_ingestion()
    
    # and then data transformation
    data_transformation = DataTransformation()
    
    
    # to save; it's taken return output from data transformation.py 
    # last one variable  is not needed because we alreay save pickle file of preprocessor.pkl
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array, test_array))
    
        
    
    
