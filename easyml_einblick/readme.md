# Easily create ML models! 
In a few quick steps, you can train ML models. 

To install, simply run two pip commands (one to get the dependencies, and one to get the functions): 
`!pip install -r https://github.com/einblick-ai/helpful-functions/raw/main/easyml-einblick/requirements.txt`



First, instantiate the ML object: 
`ml = automllite(df,"Accepted",0.5,"regress")`

Then, trigger data preprocessing:
`ml.preprocess()`

You can then start model training:
`ml.train()`

Get explainability: 
`ml.explain()`

And apply the model: 
`ml.apply_model(df2)`
