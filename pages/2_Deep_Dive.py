import streamlit

from mainfunctions import *
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from xlsxwriter import Workbook
import numpy as np

# set up error handling if no files uploaded or page refresh
if 'count' not in st.session_state:
    st.write("Please upload files on the File Upload page!")
else:
    if st.session_state["count"] == 0:
        st.write("Please upload files on the File Upload page!")

    # if file uploaded, then this loop
    if st.session_state['count'] >= 1:
        st.session_state.attributionwindowbychannel = st.session_state.attributionwindowbychannel
        # initialise df1 and df2 from file upload page
        df1 = st.session_state["df1"]
        df2 = st.session_state["df2"]

        # list of channels
        listofvalues = list(df1.medium.unique())

        # write heading
        st.markdown("<h2 style='text-align: center; color: white; font-size: Source Sans Pro;'>Channel Deep Dive</h2>", unsafe_allow_html=True)

        # initialise slider for attribution window
        st.slider(label="Set the attribution window", min_value=5, max_value=120, value=30, step=5, key="attributionwindowbychannel")
        # initialise dropdown for brand selection
        st.selectbox('What brand do you want to analyse?', listofvalues, key="medium")


        def highlight_rows(row):
            value = row.loc['P-value (%)']
            if value > 10:
                color = '#5A5A5A'  # Grey
                fontcolor = '#FFFFFF'
            else:
                color = '#0E1117'  # transparent
                fontcolor = '#FFFFFF'

            return ['background-color: {}; color: {}'.format(color, fontcolor) for r in row]

        # align right but don't think it works
        def right_align(s, props='text-align: right;'):
            return props
        print("MADE IT THIS FAR")
        t1 = AllMedium(df1, df2, st.session_state.attributionwindowbychannel, "space")
        t2 = AllMedium(df1, df2, st.session_state.attributionwindowbychannel, "ad_language")
        t3 = AllMedium(df1, df2, st.session_state.attributionwindowbychannel, "tv_program_name")
        t4 = AllMedium(df1, df2, st.session_state.attributionwindowbychannel, "tv_spot_start_time_hour")
        t5 = AllMedium(df1, df2, st.session_state.attributionwindowbychannel, "spot_position")

        st.session_state["Spot Length"] = PrintTables(t1, st.session_state.medium)
        st.session_state["Ad Language"] = PrintTables(t2, st.session_state.medium)
        st.session_state["TV Program Name"] = PrintTables(t3, st.session_state.medium)
        st.session_state["Spot Hour"] = PrintTables(t4, st.session_state.medium)
        st.session_state["Spot Position"] = PrintTables(t5, st.session_state.medium)


        def to_excel(x):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            for i in range(len(x)):
                st.session_state[x[i]]["Attribution Window"] = st.session_state.attributionwindowbychannel
                st.session_state[x[i]] = st.session_state[x[i]].drop("Power", axis=1)
                st.session_state[x[i]]["Cost Per Unit Lift ($)"] = st.session_state[x[i]]["Cost Per Unit Lift ($)"].round(2)
                st.session_state[x[i]].to_excel(writer, index=False, sheet_name=x[i])
                worksheet = writer.sheets[x[i]]
                format1 = writer.book.add_format({'num_format': '0.00'})
                worksheet.set_column('A:A', None, format1)
                formatter = writer.book.add_format({'text_wrap': True, 'valign': 'top'})
                for idx, col in enumerate(st.session_state[x[i]].columns):
                    max_len = max(st.session_state[x[i]][col].astype(str).map(len).max(), len(col))
                    worksheet.set_column(idx, idx, max_len + 2, formatter)
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        x = ["Spot Length", "Ad Language", "TV Program Name", "Spot Hour", "Spot Position"]
        df_xlsx = to_excel(x)

        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)
        with col12:
            st.download_button(label='📥 Export To Excel',
                               data=df_xlsx,
                               file_name='DeepDive.xlsx')

        # # call StatsTable function and feed it df1, df2, the medium and attribution window selected and hardcode the dimensions
        # tableSpace = StatsTable2(df1, df2, st.session_state.medium, st.session_state.attributionwindow, "space")
        # # create tables for performance by each dimension
        # # rename Dimension to the actual dimension used
        # # format column as string
        # tableSpace = tableSpace.rename({'Dimension': 'Spot Length (s)'}, axis='columns')
        # tableSpace["Spot Length (s)"] = tableSpace["Spot Length (s)"].astype(str)
        # # get rid of decimal place
        # if len(tableSpace["Spot Length (s)"][0]) > 2:
        #     tableSpace["Spot Length (s)"] = tableSpace["Spot Length (s)"].str[:-2]
        #
        # st.write("LANGUAGE")
        # tableLanguage = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "ad_language")
        # tableLanguage = tableLanguage.rename({'Dimension': 'Language'}, axis='columns')
        # tableLanguage["Language"] = tableLanguage["Language"].astype(str)
        # st.write("TV PROGRAM")
        # tableProgram = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "tv_program_name")
        # tableProgram = tableProgram.rename({'Dimension': 'TV Program'}, axis='columns')
        # tableProgram["TV Program"] = tableProgram["TV Program"].astype(str)
        #
        # tableSpotHour = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "tv_spot_start_time_hour")
        # tableSpotHour = tableSpotHour.rename({'Dimension': 'Spot Hour'}, axis='columns')
        # tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].astype(str)
        # if len(tableSpotHour["Spot Hour"][0]) > 2:
        #     tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].str[:-2]
        #
        # tableSpotPosition = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "spot_position")
        # tableSpotPosition = tableSpotPosition.rename({'Dimension': 'Spot Position'}, axis='columns')
        # tableSpotPosition["Spot Position"] = tableSpotPosition["Spot Position"].astype(str)

        #
        # # two tabs, one for tables one for charts
        # tab1, tab2, tab3 = st.tabs(["Tables","Charts","Ad Level"])
        # with tab1:
        #     # define styling for rows with p value > 10 - background grey and font colour grey
        #     def highlight_rows(row):
        #         value = row.loc['P-value (%)']
        #         if value > 10:
        #             color = '#5A5A5A'  # Grey
        #             fontcolor = '#FFFFFF'
        #         else:
        #             color = '#0E1117'  # transparent
        #             fontcolor = '#FFFFFF'
        #
        #         return ['background-color: {}; color: {}'.format(color,fontcolor) for r in row]
        #
        #     # align right but don't think it works
        #     def right_align(s, props='text-align: right;'):
        #         return props
        #
        #     # make 5 columns and write insignificant note far right - formatted as below
        #     col1, col2, col3, col4, col5 = st.columns(5)
        #     with col5:
        #         st.markdown(f'<h3 style="color:#B2BEB5;font-size:11px;">{"*statistically insignificant rows are greyed out"}</h3>',
        #                     unsafe_allow_html=True)

        # subheader each table with respective title and output tables with styling
    #     st.subheader("Performance by Spot Length")
    #     st.dataframe(tableSpace.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}').applymap(right_align), use_container_width=True)
    #
    #     st.subheader("Performance by Language")
    #     st.dataframe(tableLanguage.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
    #
    #     st.subheader("Performance by TV Program")
    #     st.dataframe(tableProgram.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
    #
    #     st.subheader("Performance by Spot Hour")
    #     st.dataframe(tableSpotHour.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
    #
    #     st.subheader("Performance by Spot Position")
    #     st.dataframe(tableSpotPosition.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
    # #
        #     pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(
        #         df1, df2, st.session_state.medium, st.session_state.attributionwindow)
        #
        # # second tab ie Charts
        # with tab2:
        #     #  make 5 columns just so that the text can be to the right
        #     col1, col2, col3, col4, col5 = st.columns(5)
        #     with col5:
        #         st.markdown(f'<h3 style="color:#B2BEB5;font-size:11px;">{"*statistically insignificant variables are greyed out"}</h3>',
        #                     unsafe_allow_html=True)
        #
        #     # make column called colour so it can reference the hex code for significant vs insignificant
        #     tableSpace["Colour"] = np.where(tableSpace["P-value (%)"] < 10, '#008080', '#5A5A5A')
        #     # make % change column for labels
        #     tableSpace["% change"] = tableSpace["Average Change (%)"].astype(str) + "%"
        #
        #     tableLanguage["Colour"] = np.where(tableLanguage["P-value (%)"] < 10, '#008080', '#5A5A5A')
        #     tableLanguage["% change"] = tableLanguage["Average Change (%)"].astype(str) + "%"
        #
        #     tableProgram["Colour"] = np.where(tableProgram["P-value (%)"] < 10, '#008080', '#5A5A5A')
        #     tableProgram["% change"] = tableProgram["Average Change (%)"].astype(str) + "%"
        #
        #     tableSpotHour["Colour"] = np.where(tableSpotHour["P-value (%)"] < 10, '#008080', '#5A5A5A')
        #     tableSpotHour["% change"] = tableSpotHour["Average Change (%)"].astype(str) + "%"
        #
        #     tableSpotPosition["Colour"] = np.where(tableSpotPosition["P-value (%)"] < 10, '#008080', '#5A5A5A')
        #     tableSpotPosition["% change"] = tableSpotPosition["Average Change (%)"].astype(str) + "%"
        #
        #
        #     # plot all the graphs
        #     PlottingDeepDive(tableSpace, "Spot Length (s)")
        #     PlottingDeepDive(tableLanguage, "Language")
        #     PlottingDeepDive(tableProgram, "TV Program")
        #     PlottingDeepDive(tableSpotHour, "Spot Hour")
        #     PlottingDeepDive(tableSpotPosition, "Spot Position")
        #
        #
        # with tab3:
        #     # chart the uplift per ad
        #     chart = pd.DataFrame(
        #         {'Before Ad': beforeAd, 'After Ad': afterAd, 'Date': date, 'Program': program, 'Language': language,
        #          'Spot Length': spotlength, 'Spot Hour': spothour, 'Spot Position': spotposition})
        #     # calculate uplift
        #     chart["Uplift %"] = round((((chart["After Ad"] / chart["Before Ad"]) - 1) * 100), 1)
        #     # get index
        #     chart = chart.reset_index()
        #
        #     col = st.radio('Pick how you want to color by', ['Program', 'Language', 'Spot Length', 'Spot Hour', 'Spot Position'])
        #
        #     fig = px.bar(chart, x='index', y='Uplift %', color=col,
        #                  hover_data={'Uplift %': True, 'index': False, 'Date': True, 'Program': True,
        #                              'Language': True, 'Spot Length': True, 'Spot Hour': True,
        #                              'Spot Position': True}
        #                  )
        #     # change layout to hide axes, initialise second y axis
        #     fig.update_layout(xaxis=dict(showgrid=False),
        #                       yaxis=dict(showgrid=False))
        #     # formatting changes, title etc
        #     fig.update_xaxes(title_font=dict(size=16, family='Source Sans Pro', color='white'))
        #     fig.update_yaxes(visible=True, showticklabels=True)
        #     fig.update_layout(
        #         title={
        #             'text': "Uplift Per Ad ",
        #             'y': 1,
        #             'x': 0.5,
        #             'xanchor': 'center',
        #             'yanchor': 'top'})
        #     # change colour based on significance
        #     fig.update_xaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
        #     fig.update_yaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
        #     st.plotly_chart(fig, use_container_width=True)

    # df1 = df1[["country", "brand","tv_program_name", "tv_spot_start_time_hour", "ad_language", "medium", "space", "amount", "spot_position", "DateJoin"]]




