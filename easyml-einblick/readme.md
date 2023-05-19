# Easily create ML models! 
In a few quick steps, you can train ML models. 

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
