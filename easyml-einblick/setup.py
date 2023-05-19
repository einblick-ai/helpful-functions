from setuptools import setup

setup(
    name='easyml-einblick',
    version='1.0',
    py_modules=['easyml-einblick'],
    install_requires=[
        'tpot','xgboost','shap'  # Add any dependencies required by your class
    ],
)
