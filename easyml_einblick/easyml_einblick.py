class easyml_einblick:
    
    def __init__(self, df, target, search_time, isClassifyOrRegression, metric = None):
        """
        This constructor creates:
        - X: Input DataFrame containing the features.
        - y: Input Series containing the target variable.
        - preprocessing_pipeline: Preprocessing pipeline to transform input features.
        - X_p: Transformed DataFrame after applying the preprocessing pipeline.
        """
        self.X = df.drop(target, axis = 1)
        self.y = df[target]
        self.isClassifyOrRegression = isClassifyOrRegression
        self.search_time = search_time 
        self.target = target 
        self.metric = metric 
    
    
    #######
    """
        The main methods you might want to call
    """
    ######
    
    def set_model(self, model, fit_model): 
        self.model = model
        self.fit_model = fit_model 
    
    
    def preprocess(self): 
        self.preprocessing_pipeline, self.X_p = self.create_preprocessing_pipeline(self.X)
    
    def train(self): 
        """
        Train the model based on the specified problem type.
        Parameters: isClassifyOrRegression: String indicating the problem type ("classification" or "regression").
        Returns: fit_model: Trained model object.
        """   
        if self.isClassifyOrRegression.startswith("class"):
            if self.metric == None: 
                self.metric = 'accuracy'
            self.model, self.fit_model = self.train_classifier(self.X_p, self.y, self.search_time, self.metric)
        elif self.isClassifyOrRegression.startswith("regress"):
            if self.metric == None: 
                self.metric = 'neg_mean_absolute_error'
            self.model, self.fit_model = self.train_regressor(self.X_p, self.y, self.search_time, self.metric)
        else:
            raise ValueError("Invalid problem type. Must select 'classification' or 'regression'")
    
    def apply_model(self, df_new): 
        """
        Applies the fit model to a new DataFrame
        """
        #try: 
        X_new = df_new.drop(self.target, axis = 1, errors='ignore')
        X_p_new = self.fit_preprocessing_pipeline.transform(X_new)
        y_new = self.fit_model.predict(X_p_new)
        
        X_new[self.target] = y_new
        return X_new          
    
        #except: 
        #    raise NameError(f"The model does not exist. Try running train() first")
        #    return -1

    #######
    """
        A few methods to access the models and pipelines
    """
    ######
    def get_preprocessing_pipeline(self): 
        try: 
            return self.preprocessing_pipeline 
        except: 
            raise NameError(f"The preprocessing pipeline does not exist. Try running preprocess() first")

    def get_model(self): 
        try: 
            return self.model 
        except: 
            raise NameError(f"The model does not exist. Try running train() first")

    def get_fit_model(self): 
        try: 
            return self.fit_model 
        except: 
            raise NameError(f"The model does not exist. Try running train() first")
         
    def explain(self): 
        """
        Runs and displays a Shap bar chart 
        """
        try: 
            self.displayShap(self.X_p, self.model)
        except: 
            raise NameError(f"The model does not exist. Try running train() first")
            
    #######
    """
        Other helper functions 
    """
    ######
    def create_preprocessing_pipeline(self, df):
        """
        Get the preprocessing pipeline for the input DataFrame.

        Inputs: Input DataFrame containing the data.

        Returns:
        preprocessing_pipeline: Preprocessing pipeline to transform the input data.
        transformed_df: Transformed DataFrame after applying the preprocessing pipeline.
        """

        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.compose import ColumnTransformer
        import pandas as pd
        from sklearn.base import BaseEstimator, TransformerMixin

        numeric_cols = df.select_dtypes(include=['int', 'float','bool']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        string_cols = df.select_dtypes(include=['object','string']).columns.tolist()

        # Define the preprocessing steps for each column type
        text_preprocessing = Pipeline([
            ('fill_nulls', SimpleImputer(strategy='constant', fill_value='null')),
            ('label_encoding', OrdinalEncoder())
        ])

        numeric_preprocessing = Pipeline([
            ('imputation', SimpleImputer(strategy='mean'))
        ])

        class DateTimeToFloatTransformer(BaseEstimator, TransformerMixin):
            def __init__(self):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X.astype(int)

        datetime_preprocessing = Pipeline([
            ('to_int', DateTimeToFloatTransformer())
        ])

        # Create the ColumnTransformer to apply different preprocessing steps to different column types
        column_transformer = ColumnTransformer([
            ('text', text_preprocessing, string_cols),
            ('numeric', numeric_preprocessing, numeric_cols),
            ('datetime', datetime_preprocessing, date_cols)
        ])

        # Create the preprocessing pipeline
        preprocessing_pipeline = Pipeline([
            ('preprocessing', column_transformer)
        ])
        
        self.fit_preprocessing_pipeline = preprocessing_pipeline.fit(df)

        return self.fit_preprocessing_pipeline, pd.DataFrame(preprocessing_pipeline.fit_transform(df), columns = df.columns)
        
    def train_classifier(self, X, y, search_time = 1, scoring = 'accuracy'):
        from tpot import TPOTClassifier
        import xgboost 
        from xgboost import XGBClassifier
        config_dict = {
            'xgboost.XGBClassifier': {
                'n_estimators': range(50, 500, 50),
                'max_depth': range(3, 10),
                'learning_rate': [0.1, 0.01, 0.001],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 1, 5],
            }
        }
        print('Beginning model training! /n')
        tpot = TPOTClassifier(generations=5, population_size=50, config_dict = config_dict,  max_time_mins=search_time, verbosity=0, scoring=scoring, random_state=42)

        # Fit TPOT on the training data
        tpot.fit(X, y)

        # Get the best pipeline found by TPOT
        best_pipeline = tpot.fitted_pipeline_
        print(tpot.score(X, y))
        
        model = best_pipeline['xgbclassifier']
        fit_model = model.fit(X,y)
        return model, fit_model  

    def train_regressor(self, X, y, search_time = 1, scoring='neg_mean_absolute_error'):
        from tpot import TPOTClassifier
        import xgboost 
        
        from xgboost import XGBRegressor
        config_dict = {
            'xgboost.XGBRegressor': {
                'n_estimators': range(50, 500, 50),
                'max_depth': range(3, 10),
                'learning_rate': [0.1, 0.01, 0.001],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 1, 5],
            }
        }

        print('Beginning model training! /n')
        tpot = TPOTClassifier(generations=5, population_size=50, config_dict = config_dict,  max_time_mins=search_time, verbosity=0, scoring=scoring, random_state=42)

        # Fit TPOT on the training data
        tpot.fit(X, y)

        # Get the best pipeline found by TPOT
        best_pipeline = tpot.fitted_pipeline_
        print(tpot.score(X, y))
        
        model = best_pipeline['xgbregressor']
        fit_model = model.fit(X,y)
        return model, fit_model  
         
    def displayShap(self, X, model):
        import shap
        if len(X) > 100:
            X_sample = X.sample(100)
        else:
            X_sample = X
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=50)
