import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models



# same kind of class variable we are creating
""" providing the input varible path providing that data coming for data
    data transfromation
"""

@dataclass
class DataTransformationconfig:
    # preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    preprocessor_obj_file_path: str = os.path.join('artifact','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()


    def get_data_transformation_object(self):
        try:
            numerical_column = ["writing_score", "reading_score"]
            categorical_column = ['gender',
                                'race_ethnicity', 
                                'parental_level_of_education',
                                'lunch', 
                                'test_preparation_course']
            
            numeric_pipeline =  Pipeline(
                                steps = [
                                        ('Imputer', SimpleImputer(strategy = 'median')),
                                        ('scaler', StandardScaler())
                                        ]
                                        )
            categorical_pipeline = Pipeline(
                                steps = [
                                        ('Imputer', SimpleImputer(strategy = 'most_frequent')),
                                        ('one_hot_encoder', OneHotEncoder()),
                                        ('scaler', StandardScaler(with_mean=False)),
                                        ]
                                        )
            
            logging.info(f"categorical_columns : {categorical_column}")
            logging.info(f"numerical_columns : {numerical_column}")
            
            
            # this variable creates to apply column transformation numeric pipeline
            # on numerical columns same as categoical pipeline on categorical column
            
            preprocessor = ColumnTransformer(
                            [
                            ("numeric_pipeline",numeric_pipeline,numerical_column),
                            ("categorical_pipeline",categorical_pipeline,categorical_column)
                            ]
                                            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
    
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
                # this reading from the data ingestion
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                
                logging.info(f"read train and test data completed")
                
                logging.info(f'obtaining preprocess object')
                
                preprocessing_obj = self.get_data_transformation_object()
                
                
                
                # seprating target columns
                target_column_name = 'math_score' 
                numerical_columns = ['writing_score','reading_score']
                # numerical_column = ['writing_score','reading_score']
                
                
                # dropping target column i.e creating x,y variables like in jupyter notebook
                input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
                target_feature_train_df = train_df[target_column_name]
                
                
                
                input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
                target_feature_test_df = test_df[target_column_name]
                
                
                
                
                logging.info(f"applying preprocessing object on training dataframe and testing dataframe")
                
                
                
                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
                
                

                
                train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
                
                logging.info(f"Saved preprocessing object.")
                
                
                # we are not save pickle file anyhwere so we providing path of picle file 
                # to save and write down the function to save in utils
                
                save_object(
                    file_path = self.data_transformation_config.preprocessor_obj_file_path,
                    obj = preprocessing_obj)
                
                
                
                # we are returning train_arr, testarr,data transformation and 
                # pickle file path 
                
                
                return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
                
                
        except Exception as e:
            raise CustomException(e, sys)

        
        
    
        
    
