import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models
from dataclasses import dataclass

""" 
importing all the regression algotithms
"""
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                BayesianRidge, HuberRegressor,
                                PassiveAggressiveRegressor, SGDRegressor)



from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, 
                            ExtraTreesRegressor, GradientBoostingRegressor, 
                            RandomForestRegressor, VotingRegressor)

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            
            
            
            
            models ={
                
                "LinearRegression" : LinearRegression(), 
                "RidgeRegression":Ridge(), 
                "LassoRegression": Lasso(), 
                "ElasticNet":ElasticNet(), 
                "BayesianRidge":BayesianRidge(), 
                "HuberRegressor":HuberRegressor(),
                "PassiveAggressiveRegressor":PassiveAggressiveRegressor(), 
                "SGDRegressor":SGDRegressor(),
                
                "AdaBoostRegressor":AdaBoostRegressor(), 
                "BaggingRegressor":BaggingRegressor(), 
                "ExtraTreesRegressor":ExtraTreesRegressor(), 
                "GradientBoostingRegressor": GradientBoostingRegressor(),

                "RandomForestRegressor": RandomForestRegressor(), 
                
                
                "DecisionTreeRegressor":DecisionTreeRegressor(), 
                "ExtraTreeRegressor": ExtraTreeRegressor(),
                
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "SVR":SVR(),
                "MLPRegressor":MLPRegressor()
                
                }
            
            params={
                "DecisionTreeRegressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForestRegressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "RidgeRegression":{}, 
                "LassoRegression": {}, 
                "ElasticNet":{}, 
                "BayesianRidge":{}, 
                "HuberRegressor":{},
                "PassiveAggressiveRegressor":{}, 
                "SGDRegressor":{},
                
                "BaggingRegressor":{}, 
                "ExtraTreesRegressor":{},
                
                "ExtraTreeRegressor": {},
                
                "KNeighborsRegressor":{},
                
                "SVR":{},
                "MLPRegressor":{},
                
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256],
                    
                }
                
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models,param=params)
            
            
            
            # to get best models score
            best_model_score = max(sorted(model_report.values()))
            
            
            # to get best model name keys we used nested list to get it
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            
            
            best_model = models[best_model_name]
            
            
            # put the thresold of 0.6
            if best_model_score < 0.6:
                raise CustomException("no best model found")
            
            logging.info(f"best model found on both train and test dataset")
            
            # dumping the pickle file as a name of model.pickle 
            # save_object is an function created inside the utils.py given dir and
            # save pickle file
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
                )
            
            
            # lets see predicted output to the test data
            
            predicted = best_model.predict(x_test)
            
            r2_square = r2_score(y_test, predicted)
            logging.info('r2_score sucessfully initiated')
            
            return r2_square,best_model_name
        
            
        except Exception as e:
            raise CustomException(e, sys)
            