# Easily create ML models! 
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
