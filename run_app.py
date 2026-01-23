import os
import sys

# Set environment variable to skip the email prompt
os.environ['STREAMLIT_EMAIL'] = ''

# Run streamlit
if __name__ == "__main__":
    os.system(f'"{sys.executable}" -m streamlit run streamlit_app.py')