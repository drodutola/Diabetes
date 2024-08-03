import os
import streamlit as st

# Get the app content from the environment variable
app_content = st.secrets["APP_PY_CONTENT"]

# Write the content to a temporary app.py file
if app_content:
    with open('app.py', 'w') as f:
        f.write(app_content)


# Execute the app.py content
exec(open('app.py').read())
