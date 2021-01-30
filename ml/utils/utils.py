import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib
import logging
import datetime
import time
import logging

class DataHandler:
    """
        Get data from csv
    """
    def __init__(self):
        self.data = None
    def get_data(self):
        logging.info(" - - - fetch data: - - - ")
        self.data = pd.read_csv('~/project_cloud_computing/ml/earthquakes.csv') 
        logging.info( " - - - data loaded - - - \nFiles : earthquakes {}".format(self.data.shape))
    def get_process_data(self):
        self.get_data()
        print(" - - - data processed - - - ")
        
class FeatureRecipe(DataHandler):
    """
    Feature processing class
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.category = None
        self.discrete_variable = None
        self.continous_variable = None
        
    #Useless feature
    def drop_useless(self):
        """
        Drop useless column
        """

        def drop_specific_col(self):
            dropped_sepcific_col = []
            dropped_sepcific_col.append('ID')
            dropped_sepcific_col.append('Location Source')
            dropped_sepcific_col.append('Magnitude Source')
            dropped_sepcific_col.append('Magnitude Type')
            return dropped_sepcific_col
               
        def drop_nan_col_100(self):
            dropped_nan_col = []
            for (columnName, columnData) in self.data.iteritems(): 
                if(self.data[columnName].isna().all() == True):
                    dropped_nan_col.append(columnName)
            print("{} feature have 100% NaN ".format(len(dropped_nan_col)))
            return dropped_nan_col
            
        def drop_nan_col_25(df:pd.DataFrame, thresold: float):
            bf=[]
            for c in self.data.columns.to_list():
                if self.data[c].isna().sum()/self.data.shape[0] > thresold:
                    bf.append(c)
            print("{} feature have more than {} NaN ".format(len(bf),thresold))
            print('\n\n - - - features - - -  \n {}'.format(bf))
            return bf
                
        self.data = self.data.drop(drop_specific_col(self), axis=1)
        self.data = self.data.drop(drop_nan_col_100(self), axis=1)
        self.data = self.data.drop(drop_nan_col_25(self, 0.25), axis=1)
        print(self.data)
        print("- - - drop useless columns - - - ")
        
    def convert_timestamp(self):
        """
        Convert date to timestamp
        """
        timestamp = []
        for d, t in zip(self.data['Date'], self.data['Time']):
            try:
                ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S')
                timestamp.append(time.mktime(ts.timetuple()))
            except ValueError:
                timestamp.append('ValueError')

        timeStamp = pd.Series(timestamp)
        self.data['Timestamp'] = timeStamp.values

        self.data = self.data.drop(['Date', 'Time'], axis=1)
        self.data = self.data[self.data.Timestamp != 'ValueError']
        print(self.data.head())
        print("- - - convert timestamp ---")
        
    def encode_categorical_variable(self):
        """
        Convert categoricals variables to numerics variables
        """
        le = preprocessing.LabelEncoder()
        le.fit(self.data['Type'])
        self.data['Type'] = le.transform(self.data['Type'])

        le.fit(self.data['Source'])
        self.data['Source'] = le.transform(self.data['Source'])

        le.fit(self.data['Status'])
        self.data['Status'] = le.transform(self.data['Status'])
        print(self.data)
        print('- - - encoding variables - - -')
        
    def prepare_data(self):
        """
        Wrap code above
        """
        self.drop_useless()
        self.convert_timestamp()
        self.encode_categorical_variable()
        print("- - - data processed - - -")
        
        
class FeatureExtractor:
    """
    Feature Extractor class
    """    
    def __init__(self, data: pd.DataFrame):
        """
            Input : pandas.DataFrame
            Output : X_train, X_test, y_train, y_test according to sklearn.model_selection.train_test_split
        """
        
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    def extract(self):
        """
            drop useless column and set x and y
        """
        x = self.data[["Latitude", "Longitude", "Timestamp", "Source", "Status", "Type"]]
        y= self.data[["Magnitude", "Depth"]]
        return x, y
    
    def split(self, size: float):
        """
            train test split
        """
        x, y = self.extract()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, random_state=42,test_size=size)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
class ModelBuilder:
    """
        Class for train and print results of ml model 
    """
    def __init__(self, model_path: str = None, save: bool = None):
        self.model = None
        
    def train(self, X, Y, model):
        if model == "RandomForestRegressor":
            self.model = RandomForestRegressor().fit(X, Y)
        
        elif model == "LinearRegression":
            self.model = LinearRegression().fit(X, Y)
        
    def predict_test(self, X) -> np.ndarray:
        print("- - - prediction: - - -") 
        print(self.model.predict(X))
        return self.model.predict(X)
    
    def save_model(self, path:str):
        joblib.dump((self.model), '{}model.joblib'.format(path))
        print('- - - Model Saved - - -')
        pass
                    
    def print_accuracy(self, X, Y):
        print("- - - Score: - - -") 
        print(self.model.score(X, Y))
        return self.model.score(X, Y)
        pass
    
    def load_model(self):
        try:
            joblib.load()
            pass
        except:
            pass
