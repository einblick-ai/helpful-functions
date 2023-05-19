# Easily create ML models! 
## Quickstart
In a few quick steps, you can train ML models. 

```python 
## Install
!pip install git+https://github.com/einblick-ai/helpful-functions.git#subdirectory=easyml_einblick

## First, instantiate the ML object: 
ml = easyml_einblick(df,"Accepted",0.5,"regress")

## Then, trigger data preprocessing:
ml.preprocess()

## You can then start model training:
ml.train()

## Get explainability: 
ml.explain()

## And apply the model to a new dataframe: 
ml.apply_model(df2)

```

You can also quickly extract all of the pipeline components once the model is trained:

```Python
## Preprocessing steps (label encoding, date to int, etc...)
ml.get_preprocessing_pipeline()

## XGBoost model not fitted to data
ml.get_model()

## XGBoost model fitted to training data
ml.get_fit_model()

```

## Scoring 
### Classification Scoring Metrics:
* 'accuracy': Classification accuracy.
* 'f1': F1 score, the harmonic mean of precision and recall.
* 'precision': Precision, the proportion of true positives to the sum of true and false positives.
* 'recall': Recall, the proportion of true positives to the sum of true positives and false negatives.
* 'roc_auc': Area under the receiver operating characteristic curve.

### Regression Scoring Metrics:
* 'neg_mean_squared_error': Negative mean squared error.
* 'neg_mean_absolute_error': Negative mean absolute error.
* 'r2': R-squared coefficient of determination.

You can select the desired scoring metric by specifying it as the value of the scoring parameter when initializing easyml


