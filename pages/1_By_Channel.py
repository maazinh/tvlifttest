from mainfunctions import *
from io import BytesIO

if 'attributionwindowbychannel' in st.session_state:
    st.session_state.attributionwindowbychannel = st.session_state.attributionwindowbychannel

#set up error handling if no files uploaded or page refresh
if 'count' not in st.session_state:
    st.write("Please upload files on the File Upload page!")
else:
    if st.session_state["count"] == 0:
        st.write("Please upload files on the File Upload page!")

    #if file uploaded, then this loop
    if st.session_state["count"] >= 1:

        #title aligned center
        st.markdown("<h2 style='text-align: center; color: white; font-size: Source Sans Pro;'>Overall Performance</h2>", unsafe_allow_html=True)

        # initialise slider
        st.slider(label="Set the attribution window", min_value=5, max_value=120, value=30, step=5,key="attributionwindowbychannel")

        #set up two tabs, normal tables and then benchmark charts
        tab1, tab2 = st.tabs(["Performance","Benchmarking"])

        #first tab - PERFORMANCE
        with tab1:

            # blank lines padding
            st.write("")

            # initialise df1 (TVDATA) and df2 (KPIDATA) from file upload page
            df1 = st.session_state["df1"]
            df2 = st.session_state["df2"]

            # shade insignificant rows grey
            def highlight_rows(row):
                value = row.loc['P-value (%)']
                if value > 10:
                    color = '#5A5A5A'  # Grey
                    fontcolor = '#FFFFFF'
                else:
                    color = '#0E1117'  # transparent
                    fontcolor = '#FFFFFF'
                return ['background-color: {}; color: {}'.format(color, fontcolor) for r in row]

            # call Tabbing function
            tab_labels, tab_labels2, tab_labels_year = Tabbing(df1)

            # 0 is index for first month - then increment by 1 to move through sorted list of months
            month = 0
            year = 0
            # loop through month name, tab name becomes month name
            for tab in st.tabs(tab_labels_year):
                for tab in st.tabs(tab_labels2):
                    with tab:
                        # call function with month NUMBER as input - starts with earliest month
                        table = ChannelStatsTableByMonth(df1, df2, tab_labels[month], tab_labels_year[year], st.session_state.attributionwindowbychannel)

                        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                        with col7:
                            st.markdown(
                                f'<h3 style="color:#B2BEB5;font-size:10px;">{"*statistically insignificant rows (p-value > 10%) are greyed out"}</h3>',
                                unsafe_allow_html=True)
                        # output table as dataframe, loop through each month
                        st.dataframe(table.style.apply(highlight_rows, axis=1).format(
                            subset=["P-value (%)", "Confidence Interval (%)"],
                            formatter="{:.2f}").format(
                            subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"],
                            formatter='{:,.0f}').format(
                            subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True, height=(len(table) + 1) * 35 + 3)


                        def to_excel(df):
                            output = BytesIO()
                            writer = pd.ExcelWriter(output, engine='xlsxwriter')
                            df["Attribution Window"] = st.session_state.attributionwindowbychannel
                            df.to_excel(writer, index=False, sheet_name="Optimisation")
                            worksheet = writer.sheets["Optimisation"]
                            format1 = writer.book.add_format({'num_format': '0.00'})
                            worksheet.set_column('A:A', None, format1)
                            formatter = writer.book.add_format({'text_wrap': True, 'valign': 'top'})
                            for idx, col in enumerate(df.columns):
                                max_len = max(df[col].astype(str).map(len).max(), len(col))
                                worksheet.set_column(idx, idx, max_len + 2, formatter)
                            writer.save()
                            processed_data = output.getvalue()
                            return processed_data


                        df_xlsx = to_excel(table)

                        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)
                        with col12:
                            st.download_button(label='📥 Export To Excel',
                                               data=df_xlsx,
                                               file_name="ByChannel-" + str(tab_labels2[month]) + str(tab_labels_year[year]) + ".xlsx",)

                        month += 1

                        # 5 columns so that insignificant warning can be to the right

                year += 1
            # run Benchmarking function, save output as benchmarkdf and listofchannels
        #     currentbrand, benchmarkdf, listofchannels = Benchmarking(table, df1, st.session_state.attributionwindowbychannel)
        #
        # # second tab
        # with tab2:
        #
        #     # plot the benchmarking charts
        #     BenchmarkingCharts(currentbrand, listofchannels,table,benchmarkdf,st.session_state.attributionwindowbychannel)
