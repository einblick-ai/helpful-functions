# A more advanced df.describe() to help you (and us) out
## Quickstart
```python 
## Install and import
!pip install --upgrade --force-reinstall git+https://github.com/einblick-ai/helpful-functions.git#subdirectory=easydescribe
from easydescribe import dataframe_summary_markdown

## Print a markdown table of the result
print(dataframe_summary_markdown(sales_export))
```
