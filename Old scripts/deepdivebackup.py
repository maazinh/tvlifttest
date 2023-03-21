from mainfunctions import *
import numpy as np


# set up error handling if no files uploaded or page refresh
if 'count' not in st.session_state:
    st.write("Please upload files on the File Upload page!")
else:
    if st.session_state["count"] == 0:
        st.write("Please upload files on the File Upload page!")

    # if file uploaded, then this loop
    if st.session_state['count'] >= 1:

        # initialise df1 and df2 from file upload page
        df1 = st.session_state["df1"]
        df2 = st.session_state["df2"]

        # list of channels
        listofvalues = list(df1.medium.unique())

        # write heading
        st.markdown("<h2 style='text-align: center; color: white; font-size: Source Sans Pro;'>Channel Deep Dive</h2>", unsafe_allow_html=True)

        # initialise slider for attribution window
        st.slider(label="Set the attribution window", min_value=5, max_value=45, value=30, step=5, key="attributionwindow")
        # initialise dropdown for brand selection
        st.selectbox('What brand do you want to analyse?', listofvalues, key="medium")

        # # call function to run t-test
        # if st.button('Calculate'):

        # # run function

        #
        #
        # # conditions for captions
        # if float(pvalue) < 10:
        #     deltalabel = "statistically significant"
        #     color = 'normal'
        # else:
        #     deltalabel = "not statistically significant"
        #     color = 'off'
        #
        # if float(power) < 70:
        #     deltalabelpower = "weak power"
        #     colorpower = 'off'
        # else:
        #     deltalabelpower = "strong power"
        #     colorpower = 'normal'
        #
        # # get rid of arrows in metric
        # st.write(
        #     """
        #     <style>
        #     [data-testid="stMetricDelta"] svg {
        #         display: none;
        #     }
        #     </style>
        #     """,
        #     unsafe_allow_html=True,
        # )
        #
        # # store table as ChannelStatsTable results - by medium
        # #table = mainfunctions.ChannelStatsTable(df1, df2, st.session_state.attributionwindow)
        # # layout for 4 metrics
        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("Avg Visits " + str(st.session_state.attributionwindow) + " mins before ad",
        #             f'{visitsBefore:,}', )
        # col2.metric("Avg Visits " + str(st.session_state.attributionwindow) + " mins after ad", f'{visitsAfter:,}',
        #             str(avgchange) + "%")
        # col3.metric("Significance", str(pvalue) + "%", deltalabel, delta_color=color)
        # col4.metric("Statistical Power", str(power) + "%", deltalabelpower, delta_color=colorpower)
        # tab_labels = TabbingDeepDive(df1)


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
        # first=0
        # for tab in st.tabs(tab_labels):
        #     with tab:
        #         tableSpace = StatsTable(df1, df2, tab_labels[first], st.session_state.attributionwindow, "space")
        #         tableSpace = tableSpace.rename({'Dimension': 'Spot Length (s)'}, axis='columns')
        #         tableSpace["Spot Length (s)"] = tableSpace["Spot Length (s)"].astype(str)
        #
        #         tableLanguage = StatsTable(df1, df2, st.session_state.medium, st.session_state.attributionwindow,
        #                                    "ad_language")
        #         tableLanguage = tableLanguage.rename({'Dimension': 'Language'}, axis='columns')
        #         tableLanguage["Language"] = tableLanguage["Language"].astype(str)
        #
        #         tableProgram = StatsTable(df1, df2, st.session_state.medium, st.session_state.attributionwindow,
        #                                   "tv_program_name")
        #         tableProgram = tableProgram.rename({'Dimension': 'TV Program'}, axis='columns')
        #         tableProgram["TV Program"] = tableProgram["TV Program"].astype(str)
        #
        #         tableSpotHour = StatsTable(df1, df2, st.session_state.medium, st.session_state.attributionwindow,
        #                                    "tv_spot_start_time_hour")
        #         tableSpotHour = tableSpotHour.rename({'Dimension': 'Spot Hour'}, axis='columns')
        #         tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].astype(str)
        #         if len(tableSpotHour["Spot Hour"][0]) > 2:
        #             tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].str[:-2]
        #
        #         tableSpotPosition = StatsTable(df1, df2, st.session_state.medium, st.session_state.attributionwindow,
        #                                        "spot_position")
        #         tableSpotPosition = tableSpotPosition.rename({'Dimension': 'Spot Position'}, axis='columns')
        #         tableSpotPosition["Spot Position"] = tableSpotPosition["Spot Position"].astype(str)
        #
        #         st.subheader("Performance by Spot Length")
        #         st.dataframe(tableSpace.style.apply(highlight_rows, axis=1).format(
        #             subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(
        #             subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(
        #             subset=["# of Ads"], formatter='{:.0f}').applymap(right_align), use_container_width=True)
        #
        #         st.subheader("Performance by Language")
        #         st.dataframe(tableLanguage.style.apply(highlight_rows, axis=1).format(
        #             subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(
        #             subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(
        #             subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
        #
        #         st.subheader("Performance by TV Program")
        #         st.dataframe(tableProgram.style.apply(highlight_rows, axis=1).format(
        #             subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(
        #             subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(
        #             subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
        #
        #         st.subheader("Performance by Spot Hour")
        #         st.dataframe(tableSpotHour.style.apply(highlight_rows, axis=1).format(
        #             subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(
        #             subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(
        #             subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
        #
        #         st.subheader("Performance by Spot Position")
        #         st.dataframe(tableSpotPosition.style.apply(highlight_rows, axis=1).format(
        #             subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(
        #             subset=["Spend ($)", "Avg Cost Per Ad ($)", "Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(
        #             subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)
        #
        #         first+=1

        # call StatsTable function and feed it df1, df2, the medium and attribution window selected and hardcode the dimensions
        tableSpace = StatsTable2(df1, df2, st.session_state.medium, st.session_state.attributionwindow, "space")
        # create tables for performance by each dimension
        # rename Dimension to the actual dimension used
        # format column as string
        tableSpace = tableSpace.rename({'Dimension': 'Spot Length (s)'}, axis='columns')
        tableSpace["Spot Length (s)"] = tableSpace["Spot Length (s)"].astype(str)
        # get rid of decimal place
        if len(tableSpace["Spot Length (s)"][0]) > 2:
            tableSpace["Spot Length (s)"] = tableSpace["Spot Length (s)"].str[:-2]

        tableLanguage = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "ad_language")
        tableLanguage = tableLanguage.rename({'Dimension': 'Language'}, axis='columns')
        tableLanguage["Language"] = tableLanguage["Language"].astype(str)

        tableProgram = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "tv_program_name")
        tableProgram = tableProgram.rename({'Dimension': 'TV Program'}, axis='columns')
        tableProgram["TV Program"] = tableProgram["TV Program"].astype(str)

        tableSpotHour = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "tv_spot_start_time_hour")
        tableSpotHour = tableSpotHour.rename({'Dimension': 'Spot Hour'}, axis='columns')
        tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].astype(str)
        if len(tableSpotHour["Spot Hour"][0]) > 2:
            tableSpotHour["Spot Hour"] = tableSpotHour["Spot Hour"].str[:-2]

        tableSpotPosition = StatsTable2(df1, df2, st.session_state.medium,st.session_state.attributionwindow, "spot_position")
        tableSpotPosition = tableSpotPosition.rename({'Dimension': 'Spot Position'}, axis='columns')
        tableSpotPosition["Spot Position"] = tableSpotPosition["Spot Position"].astype(str)


        # two tabs, one for tables one for charts
        tab1, tab2, tab3 = st.tabs(["Tables","Charts","Ad Level"])
        with tab1:
            # define styling for rows with p value > 10 - background grey and font colour grey
            def highlight_rows(row):
                value = row.loc['P-value (%)']
                if value > 10:
                    color = '#5A5A5A'  # Grey
                    fontcolor = '#FFFFFF'
                else:
                    color = '#0E1117'  # transparent
                    fontcolor = '#FFFFFF'

                return ['background-color: {}; color: {}'.format(color,fontcolor) for r in row]

            # align right but don't think it works
            def right_align(s, props='text-align: right;'):
                return props

            # make 5 columns and write insignificant note far right - formatted as below
            col1, col2, col3, col4, col5 = st.columns(5)
            with col5:
                st.markdown(f'<h3 style="color:#B2BEB5;font-size:11px;">{"*statistically insignificant rows are greyed out"}</h3>',
                            unsafe_allow_html=True)

            # subheader each table with respective title and output tables with styling
            st.subheader("Performance by Spot Length")
            st.dataframe(tableSpace.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}').applymap(right_align), use_container_width=True)

            st.subheader("Performance by Language")
            st.dataframe(tableLanguage.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)

            st.subheader("Performance by TV Program")
            st.dataframe(tableProgram.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)

            st.subheader("Performance by Spot Hour")
            st.dataframe(tableSpotHour.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)

            st.subheader("Performance by Spot Position")
            st.dataframe(tableSpotPosition.style.apply(highlight_rows, axis=1).format(subset=["Average Change (%)", "P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["Spend ($)","Avg Cost Per Ad ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').format(subset=["# of Ads"], formatter='{:.0f}'), use_container_width=True)

            pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(
                df1, df2, st.session_state.medium, st.session_state.attributionwindow)

        # second tab ie Charts
        with tab2:
            #  make 5 columns just so that the text can be to the right
            col1, col2, col3, col4, col5 = st.columns(5)
            with col5:
                st.markdown(f'<h3 style="color:#B2BEB5;font-size:11px;">{"*statistically insignificant variables are greyed out"}</h3>',
                            unsafe_allow_html=True)

            # make column called colour so it can reference the hex code for significant vs insignificant
            tableSpace["Colour"] = np.where(tableSpace["P-value (%)"] < 10, '#008080', '#5A5A5A')
            # make % change column for labels
            tableSpace["% change"] = tableSpace["Average Change (%)"].astype(str) + "%"

            tableLanguage["Colour"] = np.where(tableLanguage["P-value (%)"] < 10, '#008080', '#5A5A5A')
            tableLanguage["% change"] = tableLanguage["Average Change (%)"].astype(str) + "%"

            tableProgram["Colour"] = np.where(tableProgram["P-value (%)"] < 10, '#008080', '#5A5A5A')
            tableProgram["% change"] = tableProgram["Average Change (%)"].astype(str) + "%"

            tableSpotHour["Colour"] = np.where(tableSpotHour["P-value (%)"] < 10, '#008080', '#5A5A5A')
            tableSpotHour["% change"] = tableSpotHour["Average Change (%)"].astype(str) + "%"

            tableSpotPosition["Colour"] = np.where(tableSpotPosition["P-value (%)"] < 10, '#008080', '#5A5A5A')
            tableSpotPosition["% change"] = tableSpotPosition["Average Change (%)"].astype(str) + "%"


            # plot all the graphs
            PlottingDeepDive(tableSpace, "Spot Length (s)")
            PlottingDeepDive(tableLanguage, "Language")
            PlottingDeepDive(tableProgram, "TV Program")
            PlottingDeepDive(tableSpotHour, "Spot Hour")
            PlottingDeepDive(tableSpotPosition, "Spot Position")


        with tab3:
            # chart the uplift per ad
            chart = pd.DataFrame(
                {'Before Ad': beforeAd, 'After Ad': afterAd, 'Date': date, 'Program': program, 'Language': language,
                 'Spot Length': spotlength, 'Spot Hour': spothour, 'Spot Position': spotposition})
            # calculate uplift
            chart["Uplift %"] = round((((chart["After Ad"] / chart["Before Ad"]) - 1) * 100), 1)
            # get index
            chart = chart.reset_index()

            col = st.radio('Pick how you want to color by', ['Program', 'Language', 'Spot Length', 'Spot Hour', 'Spot Position'])

            fig = px.bar(chart, x='index', y='Uplift %', color=col,
                         hover_data={'Uplift %': True, 'index': False, 'Date': True, 'Program': True,
                                     'Language': True, 'Spot Length': True, 'Spot Hour': True,
                                     'Spot Position': True}
                         )
            # change layout to hide axes, initialise second y axis
            fig.update_layout(xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=False))
            # formatting changes, title etc
            fig.update_xaxes(title_font=dict(size=16, family='Source Sans Pro', color='white'))
            fig.update_yaxes(visible=True, showticklabels=True)
            fig.update_layout(
                title={
                    'text': "Uplift Per Ad ",
                    'y': 1,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            # change colour based on significance
            fig.update_xaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
            fig.update_yaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
            st.plotly_chart(fig, use_container_width=True)

    # df1 = df1[["country", "brand","tv_program_name", "tv_spot_start_time_hour", "ad_language", "medium", "space", "amount", "spot_position", "DateJoin"]]




