import streamlit as st
import pandas as pd
import numpy as np
import mainfunctions
from mainfunctions import *
from streamlit_extras.switch_page_button import switch_page
import time

#make page widescreen
st.set_page_config(layout="wide")

st.markdown("<h2 style='text-align: center; color: white; font-size: Source Sans Pro;'>File Upload</h2>", unsafe_allow_html=True)
st.write("")
#create empty placeholder
placeholder = st.empty()

#initialise count
if 'count' not in st.session_state:
    st.session_state['count'] = 0

#if count = 0 ie no files uploaded, show file upload boxes
if st.session_state['count'] == 0:


    with placeholder.container():

        #first file upload box
        file1 = st.file_uploader(label="Upload TV Data as a CSV file", key="first")

        #if file uploaded, read csv - seek is buffer to fix dumb issue
        if file1 is not None:
            file1.seek(0)
            st.session_state["file1"] = pd.read_csv(file1, parse_dates=["c_trans_date"], dayfirst=True)


        if "file1" in st.session_state:
            file1.seek(0)
            st.session_state["file1"] = pd.read_csv(file1, parse_dates=["c_trans_date"], dayfirst=True)

        #second file upload box
        file2 = st.file_uploader(label="Upload KPI Data as a CSV file", key="second")

        # if file uploaded, read csv - seek is buffer to fix dumb issue
        if file2 is not None:
            file2.seek(0)
            st.session_state["file2"] = pd.read_csv(file2)

        if "file2" in st.session_state:
            file2.seek(0)
            st.session_state["file2"] = pd.read_csv(file2)

        #if upload is clicked, increase count by 1, convert columns to int, initialise df1 and df2 as session variables and process them and THEN switch channel
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18 = st.columns(18)
        with col18:
            st.write("")
            if st.button('Upload'):
                with st.spinner("Processing"):
                    my_bar = st.progress(0)

                    st.session_state['count'] += 1
                    # initialise df1 and df2 and process them with functions
                    tvdata = st.session_state["file1"]
                    tvdata.tv_spot_start_time_second = tvdata.tv_spot_start_time_second.astype(int)
                    tvdata.tv_spot_start_time_minute = tvdata.tv_spot_start_time_minute.astype(int)
                    tvdata.tv_spot_start_time_hour = tvdata.tv_spot_start_time_hour.astype(int)
                    st.session_state["df1"] = mainfunctions.ProcessTVData(tvdata)
                    kpidata = st.session_state["file2"]
                    st.session_state["df2"] = mainfunctions.ProcessKPIData(kpidata)
                    switch_page("by channel")


#if files are uploaded, tell them thank you
if st.session_state['count'] >= 1:
    st.write("Thank you for uploading your files")
    st.write("To reupload files, please refresh the browser page")