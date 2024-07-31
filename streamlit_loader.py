import os
import streamlit as st

# Get the app content from the environment variable
app_content = os.environ.get('APP_PY_CONTENT')

# Write the content to a temporary app.py file
if app_content:
    with open('app.py', 'w') as f:
        f.write(app_content)

# Now you can import and use your app as usual
from app import your_main_function

# Run your main function
your_main_function()
