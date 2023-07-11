
def dataframe_summary(df):    
    import pandas as pd
    import numpy as np
    summary = pd.DataFrame()
    summary['Column'] = df.columns
    summary['Data Type'] = df.dtypes.astype(str).values
    summary['Unique Values'] = df.nunique().values
    summary['Missing Values'] = df.isnull().sum().values
    
    result_series = pd.Series()
    # Loop through each column in the dataframe
    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype != 'object':
            # Calculate the mean of the column
            mean_value = df[column].mean()
            # Add the mean value to the series with column name as index
            result_series[column] = mean_value
        else:
            # Add NaN to the series with column name as index
            result_series[column] = np.NaN
        
    summary['Mean'] = result_series.values
    summary['Min'] = df.min().values 
    summary['Max'] = df.max().values 
    result_series = df.apply(lambda x: x.value_counts().index[:3]).values 
    arr_str = np.array([', '.join(map(str, el)) for el in result_series], dtype='object')
    summary['Top 3 Common'] = arr_str
    summary['90th Percentile'] = df.apply(lambda x: np.percentile(x.astype(float), 90) if x.dtype != 'object' else np.nan).values 
    summary['10th Percentile'] = df.apply(lambda x: np.percentile(x.astype(float), 10) if x.dtype != 'object' else np.nan).values

    return summary

def dataframe_summary_markdown(df):    
    import pandas as pd
    import numpy as np
    summary = pd.DataFrame()
    summary['Column'] = df.columns
    summary['Data Type'] = df.dtypes.astype(str).values
    summary['Unique Values'] = df.nunique().values
    summary['Missing Values'] = df.isnull().sum().values
    
    result_series = pd.Series()
    # Loop through each column in the dataframe
    for column in df.columns:
        # Check if the column is numeric
        if df[column].dtype != 'object':
            # Calculate the mean of the column
            mean_value = df[column].mean()
            # Add the mean value to the series with column name as index
            result_series[column] = mean_value
        else:
            # Add NaN to the series with column name as index
            result_series[column] = np.NaN
        
    summary['Mean'] = result_series.values
    summary['Min'] = df.min().values 
    summary['Max'] = df.max().values 
    result_series = df.apply(lambda x: x.value_counts().index[:3]).values 
    arr_str = np.array([', '.join(map(str, el)) for el in result_series], dtype='object')
    summary['Top 3 Common'] = arr_str
    summary['90th Percentile'] = df.apply(lambda x: np.percentile(x.astype(int), 90) if x.dtype != 'object' else np.nan).values 
    summary['10th Percentile'] = df.apply(lambda x: np.percentile(x.astype(int), 10) if x.dtype != 'object' else np.nan).values

    return summary.to_markdown(index=False)
