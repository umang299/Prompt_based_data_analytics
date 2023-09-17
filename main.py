import pandas as pd
import pandasai as pdai
from pandasai.llm import OpenAI

import streamlit as st


def load_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=0)
    df.reset_index(inplace=True)
    return df

if __name__ == '__main__':
    api_key = st.text_input(label="OpenAI API KEY")
    if api_key != '':
        llm = OpenAI(api_token=api_key)
        csv_file = st.file_uploader(label='Upload excel file.', type=['csv'])
        if csv_file is not None:
            df = load_df(csv_path=csv_file)
            smart_df = pdai.SmartDataframe(df=df, config={"llm": llm})
        
            tab1, tab2, tab3 = st.tabs(tabs=['Info', 'Chat', 'Table'])
            with tab2:
                input = st.text_input(label="Text query")
                response = smart_df.chat(query=input)
                if input is not None:
                    st.text_area(label='Response', value=response)

            with tab1:
                st.header("Data Description")
                st.table(df.describe())
            with tab3:
                st.table(df.head(10))
    else:
        pass
    
    