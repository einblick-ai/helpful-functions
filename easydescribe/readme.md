!pip install --upgrade --force-reinstall git+https://github.com/einblick-ai/helpful-functions.git#subdirectory=easydescribe
from easydescribe import dataframe_summary_markdown
print(dataframe_summary_markdown(sales_export))
