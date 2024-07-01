import os
import shutil
import pandas as pd
import pandasai as pdai
from pandasai.llm import OpenAI

import streamlit as st
from streamlit.file_util import get_streamlit_file_path

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

credential_path = get_streamlit_file_path("credentials.toml")
if not os.path.exists(credential_path):
    os.makedirs(os.path.dirname(credential_path), exist_ok=True)
    shutil.copyfile(os.path.join(PROJECT_ROOT, ".streamlit\\credentials.toml"), credential_path)


def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=0)
    df.reset_index(inplace=True)
    return df


api_key = os.getenv['OPENAI_API_KEY']
if api_key != '':
    llm = OpenAI(api_token=api_key)
    csv_file = st.file_uploader(label='Upload excel file.', type=['csv'])
    if csv_file is not None:
        df = load_df(csv_path=csv_file)
        smart_df = pdai.SmartDataframe(df=df, config={"llm": llm})

        tab1, tab2, tab3 = st.tabs(tabs=['Chat', 'Summary', 'Top 10'])
        with tab1:
            input = st.text_input(label="Text query")
            response = smart_df.chat(query=input)
            if input is not None:
                st.text_area(label='Response', value=response)

        with tab2:
            st.header("Data Description")
            st.table(df.describe())
        with tab3:
            st.table(df.head(10))
else:
    pass