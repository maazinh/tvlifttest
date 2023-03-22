import pandas as pd
import pingouin as pg
import numpy as np
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from joblib import Parallel, delayed


pd.options.display.float_format = '{:,}'.format

### ------------------------- FILE UPLOAD PAGE -----------------------------------------------

# highlight insignificant rows with grey
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

# Process TVData file
@st.cache_data(show_spinner=False)
def ProcessTVData(tvdata):
    start = time.time()
    # filter relevant columns
    tvdata = tvdata[["country", "c_trans_date", "tv_spot_start_time_hour", "tv_spot_start_time_minute", "brand", "tv_program_name", "ad_language", "medium", "space", "amount", "mm", "yy", "spot_position"]]

    # dictionary to map old hours to new hours
    mapping_dict = {1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19', 20: '20', 21: '21', 22: '22', 23: '23', 24: '00', 25: '01', 26: '02', 27: '03', 28: '04', 29: '05'}

    # pad minutes
    tvdata["tv_spot_start_time_minute"] = tvdata["tv_spot_start_time_minute"].astype(str).str.pad(2, side='left', fillchar='0')
    #tvdata.loc[:,"tv_spot_start_time_minute"] = tvdata["tv_spot_start_time_minute"].astype(str).str.pad(2, side='left', fillchar='0')

    tvdata["tv_spot_start_time_hour"] = tvdata["tv_spot_start_time_hour"].apply(lambda x: mapping_dict[x])

    #tvdata.loc[:,"tv_spot_start_time_hour"] = tvdata["tv_spot_start_time_hour"].apply(lambda x: mapping_dict[x])
    # make DateJoin column from c_trans_date as YYYY-mm-dd HH:MM
    tvdata["DateJoin"] = tvdata["c_trans_date"].apply(lambda x: x.strftime('%Y-%m-%d')) + " " + tvdata["tv_spot_start_time_hour"] + ":" + tvdata["tv_spot_start_time_minute"]
    #tvdata.loc[:,"DateJoin"] = tvdata["c_trans_date"].apply(lambda x: x.strftime('%Y-%m-%d')) + " " + tvdata["tv_spot_start_time_hour"] + ":" + tvdata["tv_spot_start_time_minute"]
    # convert DateJoin to datetime
    tvdata["DateJoin"] = pd.to_datetime(tvdata["DateJoin"], format='%Y-%m-%d %H:%M')
    #tvdata.loc[:,"DateJoin"] = pd.to_datetime(tvdata["DateJoin"], format='%Y-%m-%d %H:%M')
    # define list of countries that are GMT+3 to change timezone to match GA
    gmtplus3 = ['ksa', 'iraq', 'jordan', 'pan arab', 'Pan Arab', 'KSA', 'JORDAN', 'IRAQ', 'PAN ARAB', 'Ksa', 'Jordan',
                'Iraq']

    # if row is from GMT+3 countries, then add one hour to make UAE timezone for GA
    tvdata["DateJoin"] = np.where(np.isin(tvdata["country"],gmtplus3), tvdata["DateJoin"] + pd.Timedelta(hours=1), tvdata["DateJoin"])
    #tvdata.loc[:,"DateJoin"] = np.where(np.isin(tvdata["country"],gmtplus3), tvdata["DateJoin"] + pd.Timedelta(hours=1), tvdata["DateJoin"])
    # also add hour to spot hour so the Deep Dive tables are accurate
    tvdata["tv_spot_start_time_hour"] = np.where(np.isin(tvdata["country"],gmtplus3), tvdata["tv_spot_start_time_hour"].astype(int) + 1, tvdata["tv_spot_start_time_hour"].astype(int))
    #tvdata.loc[:,"tv_spot_start_time_hour"] = np.where(np.isin(tvdata["country"],gmtplus3), tvdata["tv_spot_start_time_hour"].astype(int) + 1, tvdata["tv_spot_start_time_hour"].astype(int))
    # define list of hours that are from the next day
    nextdayhours = [0, 1, 2, 3, 4, 5]

    # add day to DateJoin for those rows where hour is 0-5
    tvdata["DateJoin"] = np.where(np.isin(tvdata["tv_spot_start_time_hour"],nextdayhours), tvdata["DateJoin"] + pd.Timedelta(days=1), tvdata["DateJoin"])
    #tvdata.loc[:,"DateJoin"] = np.where(np.isin(tvdata["tv_spot_start_time_hour"],nextdayhours), tvdata["DateJoin"] + pd.Timedelta(days=1), tvdata["DateJoin"])
    # return as df1
    df1 = tvdata
    df1 = df1[["country", "tv_spot_start_time_hour", "brand", "tv_program_name", "ad_language", "medium", "space", "amount", "mm", "yy", "spot_position", "DateJoin"]]

    end = time.time()


    return df1

# Process KPIData file
@st.cache_data(show_spinner=False)
def ProcessKPIData(kpidata):
    start = time.time()
    # clean GA data, get rid of thousands separators
    if type(kpidata["Pageviews"][0]) == str:
        kpidata["Pageviews"] = kpidata["Pageviews"].str.replace(",", "").astype(int)

    #kpidata["Sessions"] = kpidata["Sessions"].str.replace(",", "").astype(int)

    # change date to same format as Statex file and convert to datetime object
    kpidata["DateJoin"] = kpidata["Date"].astype(str).str[0:4] + "-" + kpidata["Date"].astype(str).str[4:6] + "-" + kpidata["Date"].astype(str).str[6:8] + " " + kpidata["Hour"].astype(str) + ":" + kpidata["Minute"].astype(str)
    kpidata["DateJoin"] = pd.to_datetime(kpidata["DateJoin"], format=('%Y-%m-%d %H:%M'))


    # create new dataframe, full range of dates with minimum being earliest and max being latest date of kpidata
    fulldates = pd.DataFrame(pd.date_range(start=min(kpidata["DateJoin"]), end=max(kpidata["DateJoin"]),freq='min'),columns=['DateJoin']).sort_values(by="DateJoin").reset_index(drop=True)

    # join kpidata onto full date range dataframe
    fulldates = pd.merge(fulldates,kpidata,how='left',on="DateJoin")

    # only keep datejoin and pageviews columns
    fulldates = fulldates[["DateJoin","Pageviews","Month"]]

    # fill blank cells with 0
    fulldates = fulldates.fillna(0)


    # return as df2
    df2 = fulldates
    end = time.time()
    return df2



### ------------------------- BY CHANNEL PAGE -----------------------------------------------



# Make tabs for all months and years - for By Channel page
def Tabbing(df1):
    # mapping dictionary to convert month names to month numbers
    mapping_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                    11: 'Nov', 12: 'Dec'}

    # reverse dictionary for mapping
    mapping_dict_reverse = {v: k for k, v in mapping_dict.items()}

    # unique list of month labels and sort - this is month numbers
    tab_labels = df1["mm"].unique().tolist()
    tab_labels.sort()
    tab_labels_year = df1["yy"].unique().astype(str).tolist()
    tab_labels_year.sort()
    # this is month numbers
    tab_labels2 = [mapping_dict.get(item, item) for item in tab_labels]

    return tab_labels, tab_labels2, tab_labels_year

# Calculate stats for all channels by month - for By Channel page
@st.cache_data(show_spinner=False)
def ChannelStatsTableByMonth(df1, df2, month, year, attributionwindow):

    # find list of mediums
    start = time.time()
    medium = df1["medium"].unique()

    df1Filtered = df1[df1["yy"] == int(year)]
    df1Filtered = df1Filtered[df1Filtered["mm"] == month]


    df2Filtered = df2[df2["DateJoin"] <= df1Filtered["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    df2Filtered = df2[df2["DateJoin"] >= df1Filtered["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]
    # print(df1Filtered.head())
    # print(df2Filtered.head())

    table = pd.DataFrame({'Channel': pd.Series(dtype='str'),
                          'Confidence Interval (%)': pd.Series(dtype='str'),
                    'P-value (%)': pd.Series(dtype='str'),
                    'Spend ($)': pd.Series(dtype='float'),
                    '# of Ads': pd.Series(dtype='float'),
                    'Avg Cost Per Ad ($)': pd.Series(dtype='float'),
                    'Cost Per Unit Lift ($)': pd.Series(dtype='float')})

    end = time.time()

    # initialise empty curves dictionary
    curves = {}
    # loop through each channel and run CalculateLift function
    for i in medium:
        start2 = time.time()
        # print(month)
        # print("channelstatstablePPPPPPPPPPPPOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP")
        # make sure there's more than one ad, otherwise throws error
        if len(df1Filtered[df1Filtered["medium"] == i].index) > 1:

            # run CalculateLift function and store in variables
            #pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(df1Filtered, df2Filtered, i, attributionwindow)

            end2 = time.time()
            confidenceinterval, pvalue, power, spend, numberofads = Lift(df1Filtered, df2Filtered, i, attributionwindow)
            end3 = time.time()

            # for each medium, append positive uplift and cost to curves dictionary
            #curves[i] = [prop2,amountPerAd]
            # calculate cost per lift
            if confidenceinterval > 0:
                #CPL = avgspend / avgchange
                CPL = (spend/numberofads)/confidenceinterval
            else:
                #CPL = 0
                CPL = 0
            # append to table
            #table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, count, avgspend, CPL]
            table.loc[len(table.index)] = [i, confidenceinterval, pvalue, spend, numberofads, spend/numberofads, CPL]
            end4 = time.time()


    end5 = time.time()
    table = table.sort_values(by="Confidence Interval (%)", axis=0, ascending=False)

    table = table.reset_index(drop=True)
    end6 = time.time()
    # print(end6-end5)
    # print("table sorting")

    return table


# Lift function that's called by ChannelStatsTableByMonth - for By Channel page
@st.cache_data(show_spinner=False)
def Lift(df1, df2, medium, attributionwindow):
    start = time.time()

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1["medium"] == medium]
    #end = time.time()
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)
    #end2 = time.time()

    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)
    #end3 = time.time()
    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= df2["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= df2["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]
    #end4 = time.time()
    dfFiltered["DateJoin"] = pd.to_datetime(dfFiltered["DateJoin"], format=('%Y-%m-%d %H:%M'))
    df2["DateJoin"] = pd.to_datetime(df2["DateJoin"], format=('%Y-%m-%d %H:%M'))
    #end5 = time.time()


    # left join GA to tvdata

    # joineddata = df2.merge(dfFiltered, on="DateJoin",how='left')
    dfFiltered.set_index('DateJoin', inplace=True)
    df2.set_index('DateJoin', inplace=True)
    joineddata = df2.join(dfFiltered, how='left')

    #end6 = time.time()
    #joineddata = joineddata.drop_duplicates(subset=['DateJoin'])

    joineddata["beforeAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(1)



    joineddata["afterAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(-attributionwindow+1)
    #end7 = time.time()

    # only include rows that are a match
    joineddata = joineddata[joineddata['country'].str.len() > 0]
    joineddata.loc[joineddata['beforeAd'] > 0, 'prop'] = joineddata["afterAd"]/joineddata["beforeAd"] - 1
    joineddata.loc[joineddata['beforeAd'] == 0, 'prop'] = 1
    joineddata.loc[joineddata['afterAd'] == 0, 'prop'] = 0
    #end8 = time.time()

    results = pg.ttest(joineddata["prop"], 0.0, paired=False, alternative='greater', confidence=0.90)


    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    power = float(round(results['power']*100, 2).to_string(index=False))
    confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    #avgchange = round(np.median(joineddata["prop"])*100,2)
    #avgchange = round(((sum(joineddata["afterAd"]) / sum(joineddata["beforeAd"]) - 1) * 100), 2)

    spend = joineddata["amount"].sum()
    numberofads = len(joineddata["amount"])


    return confidenceinterval, pvalue, power, spend, numberofads

# Benchmark any runs into an Excel file - for By Channel page
@st.cache_data(show_spinner=False)
# def Benchmarking(table, df1, attributionwindowbychannel):
#     # make a copy of the t test results stored in table as table2
#     table2 = table
#     # make new column for attribution window
#     table2["Attribution Window"] = attributionwindowbychannel
#     # make column for brand - the brand the TV data is for
#     table2["Brand"] = df1["brand"][1]
#     currentbrand = df1["brand"][1]
#     # table2 = table2[table2["P-value (%)"] < 10]
#
#     # make hdr a variable that makes it output header if new file otherwise if file already exists then just output data without headers
#     hdr = False if os.path.isfile(
#         'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv') else True
#     # export to benchmarks file - APPEND data
#     table2.to_csv(
#         'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv',
#         mode='a', header=hdr, index=False)
#
#     # find list of channels for that specific brand
#     listofchannels = table["Channel"].unique()
#     # create list
#     d = {'Channel': listofchannels}
#     # create dataframe with list
#     benchmarkdf = pd.DataFrame(data=d)
#     # import benchmark excel file
#     benchmarkexcel = pd.read_csv(
#         "C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv")
#     # create new df merging channel list from before with the actual benchmarks so as to get common list
#     benchmarkdf = pd.merge(benchmarkdf, benchmarkexcel, on=['Channel'])
#
#     # filter df for attribution window selected
#     benchmarkdf = benchmarkdf[benchmarkdf["Attribution Window"] == st.session_state.attributionwindowbychannel]
#     # round and drop duplicates
#     benchmarkdf = benchmarkdf.round(2)
#     benchmarkdf = benchmarkdf.drop_duplicates(subset=None,keep="first",inplace=False)
#     # reset index
#     benchmarkdf = benchmarkdf.reset_index(drop=True)
#
#     return currentbrand, benchmarkdf, listofchannels

# Plot charts for benchmarking vs industry - for By Channel page
def BenchmarkingCharts(currentbrand, listofchannels, table, benchmarkdf, attributionwindowbychannel):
    # define list of metrics we want to benchmark
    listofmetrics = ['Confidence Interval (%)', 'Avg Cost Per Ad ($)', 'Cost Per Unit Lift ($)']
    # loop through this list
    for x in listofmetrics:
        # initialise the df for charting
        chartdata = pd.DataFrame(
            {'Channel': pd.Series(dtype='str'), x: pd.Series(dtype='str'), "Metric": pd.Series(dtype='str'),
             "Benchmark": pd.Series(dtype='str')})
        # loop through channels
        blankchannels = []
        for i in listofchannels:
            # create new table that only looks at that channel
            table2 = table[table["Channel"] == i]
            # now filter benchmark data for only that channel
            benchmarkdf2 = benchmarkdf[benchmarkdf["Channel"] == i]
            # filter brand for anything except the current brand as we want industry average
            benchmarkdf2 = benchmarkdf2[benchmarkdf2["Brand"] != currentbrand]
            # round and drop duplicates
            table2 = table2.round(2)
            table2 = table2.drop_duplicates()
            # reset index
            table2 = table2.reset_index(drop=True)
            # round and drop duplicates
            benchmarkdf2 = benchmarkdf2.round()
            benchmarkdf2 = benchmarkdf2.drop_duplicates()
            # reset index
            benchmarkdf2 = benchmarkdf2.reset_index(drop=True)
            # get metric
            metric = round(float(table2[x][0]), 2)
            # get corresponding benchmark IF there exists any data
            if len(benchmarkdf2) > 0:
                benchmark = round(float(benchmarkdf2[x].mean()), 2)
            else:
                # if no data, append blank channel name to list of blankchannels and move onto next loop
                blankchannels.append(i)
                continue
            # get the difference IF Confidence Interval, then positive difference is GOOD
            if x == "Confidence Interval (%)":
                if benchmark == 0:
                    # if benchmark 0 then leave it
                    continue
                else:
                    metricvsbenchmark = round(100 * (metric/benchmark - 1), 0)
            else:
                if benchmark == 0:
                    # if benchmark 0 then leave it
                    continue
                else:
                    # because these are negative metrics, we flip sign cause negative is GOOD
                    metricvsbenchmark = -1 * round(100 * (metric / benchmark - 1), 0)
            # append result for each channel to chartdata df to plot along with the metric and the benchmark
            chartdata.loc[len(chartdata.index)] = [i, metricvsbenchmark, metric, benchmark]

        # if all channels are blank, write no industry data
        if len(blankchannels) == len(listofchannels):
            st.write("No industry data for " + x)
        else:
            # make colour column for pos and neg
            chartdata["Colour"] = np.where(chartdata[x] < 0, '#5A5A5A', '#008080')
            # make string column for change with % sign
            chartdata["% change"] = chartdata[x].astype(str) + "%"
            # initialise plotly bar chart
            fig = px.bar(chartdata, x="Channel", y=x, hover_data={x: False, 'Metric': True, 'Benchmark':True},
                         labels={'Channel': ' ', x: ' '}, text="% change")
            # hide axes (i think)
            fig.update_layout(xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=False))
            # change x axis title font
            fig.update_xaxes(title_font=dict(size=16, family='Source Sans Pro', color='white'))
            # hide y axis and tick marks
            fig.update_yaxes(visible=False, showticklabels=False)
            # change chart title
            fig.update_layout(
                title={
                    'text': x + " vs Industry Benchmark",
                    'y': 1,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            # change colour for pos vs neg
            fig.update_traces(marker_color=chartdata["Colour"])
            # make axes line thinner
            fig.update_xaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
            fig.update_yaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')
            # plot chart
            st.plotly_chart(fig, use_container_width=True)



### ------------------------- DEEP DIVE PAGE -----------------------------------------------



# Make tabs for all months and years - for By Channel page
def TabbingDeepDive(df1):
    # mapping dictionary to convert month names to month numbers
    mapping_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                    11: 'Nov', 12: 'Dec'}

    # reverse dictionary for mapping
    mapping_dict_reverse = {v: k for k, v in mapping_dict.items()}

    # unique list of month labels and sort - this is month numbers
    tab_labels = df1["medium"].unique().tolist()
    tab_labels.sort()
    # this is month numbers
    tab_labels2 = [mapping_dict.get(item, item) for item in tab_labels]

    return tab_labels

def PrintTables(table_list, medium):
    tables = []
    for j in table_list:
        tables.append(j)
        if len(j) > 0:
            if j["Medium"][0] == medium:
                if j["Name of Dimension"][0] == "space":
                    st.write("")
                    st.subheader("Performance by Spot Length")

                    # 5 columns so that insignificant warning can be to the right
                    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                    with col7:
                        st.markdown(
                            f'<h3 style="color:#B2BEB5;font-size:10px;">{"*statistically insignificant rows (p-value > 10%) are greyed out"}</h3>',
                            unsafe_allow_html=True)
                    table = pd.DataFrame(j)
                    table = table.drop(columns=["Medium","Name of Dimension","Power"]).rename({'Dimension': 'Spot Length (s)'}, axis='columns')
                    st.dataframe(table.style.apply(highlight_rows, axis=1).format(subset=["P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["# of Ads","Spend ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}').applymap(right_align), use_container_width=True)
                elif j["Name of Dimension"][0] == "ad_language":
                    st.write("")
                    st.subheader("Performance by Language")
                    st.write("")
                    table = pd.DataFrame(j)
                    table = table.drop(columns=["Medium","Name of Dimension","Power"]).rename({'Dimension': 'Language'}, axis='columns')
                    st.dataframe(table.style.apply(highlight_rows, axis=1).format(subset=["P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["# of Ads","Spend ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}'), use_container_width=True)
                elif j["Name of Dimension"][0] == "tv_program_name":
                    st.write("")
                    st.subheader("Performance by TV Program")
                    st.write("")
                    table = pd.DataFrame(j)
                    table = table.drop(columns=["Medium","Name of Dimension","Power"]).rename({'Dimension': 'TV Program'}, axis='columns')
                    #         tableSpace = tableSpace.rename({'Dimension': 'Spot Length (s)'}, axis='columns')
                    st.dataframe(table.style.apply(highlight_rows, axis=1).format(subset=["P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["# of Ads","Spend ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}'), use_container_width=True)
                elif j["Name of Dimension"][0] == "tv_spot_start_time_hour":
                    st.write("")
                    st.subheader("Performance by Spot Hour")
                    st.write("")
                    table = pd.DataFrame(j)
                    table = table.drop(columns=["Medium","Name of Dimension","Power"]).rename({'Dimension': 'Spot Hour'}, axis='columns')
                    st.dataframe(table.style.apply(highlight_rows, axis=1).format(subset=["P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["# of Ads","Spend ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}'), use_container_width=True)
                elif j["Name of Dimension"][0] == "spot_position":
                    st.write("")
                    st.subheader("Performance by Spot Position")
                    st.write("")
                    table = pd.DataFrame(j)
                    table = table.drop(columns=["Medium","Name of Dimension","Power"]).rename({'Dimension': 'Spot Position'}, axis='columns')
                    st.dataframe(table.style.apply(highlight_rows, axis=1).format(subset=["P-value (%)", "Confidence Interval (%)"], formatter="{:.2f}").format(subset=["# of Ads","Spend ($)","Cost Per Unit Lift ($)"], formatter='{:,.0f}'), use_container_width=True)
    finaltable = pd.concat(tables, ignore_index=True)
    #finaltable.to_csv('C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/FullDataFrame.csv',mode='a')

    return finaltable

@st.cache_data(show_spinner="Running intense calculations - can take a couple of minutes")
def AllMedium(df1, df2, attributionwindow, dimension):
    start = time.time()
    medium = df1["medium"].unique()

    # def filter_df(medium):
    #     return df1[df1["medium"] == medium]
    #
    # filtered_dfs = Parallel(n_jobs=-1)(
    #     delayed(filter_df)(m) for m in medium)

    table_list = Parallel(n_jobs=1)(
        delayed(StatsTable)(df1[df1["medium"] == medium[i]], df2, medium[i], attributionwindow, dimension) for i in range(len(medium)))

    # table = pd.DataFrame(table_list,
    #                      columns=['Medium', 'Name of Dimension', 'Dimension', 'P-value (%)', 'Power',
    #                               'Confidence Interval (%)', 'Average Change (%)',
    #                               'Spend ($)', '# of Ads'])

    end = time.time()
    print("times")
    print(end-start)
    return table_list

# Table of dimensions with their stats ie Performance by Dimension - for Deep Dive page
@st.cache_data(show_spinner="Running intense calculations - can take a couple of minutes")
def StatsTable(df1, df2, medium, attributionwindow, dimension):

    # dfFiltered is df filtered for specific medium, then list of unique variables for that specific dimension ie if Spot Hour then 07,08,09 etc
    dfFiltered = df1
    var = dfFiltered[dimension].unique()

    # loop through dimension and if it has more than one ad, run CalculateLift2 script
    table_list = Parallel(n_jobs=1)(
        delayed(LiftDeepDive)(df1, df2, medium, attributionwindow, i, dimension) for i in var if
        len(dfFiltered[dfFiltered[dimension] == i].index) > 1)

    table = pd.DataFrame(table_list,
                         columns=['Medium','Name of Dimension', 'Dimension', 'Confidence Interval (%)', 'P-value (%)', 'Power', 'Spend ($)', '# of Ads', "Cost Per Unit Lift ($)"])

    table = table.sort_values(by="Confidence Interval (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)

    return table

# Lift function for Deep Dive page for table by metrics
@st.cache_data(show_spinner="Running intense calculations - can take a couple of minutes")
def LiftDeepDive(df1, df2, medium, attributionwindow, i, dimension):

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1[dimension] == i]
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)

    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= df2["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= df2["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]

    dfFiltered["DateJoin"] = pd.to_datetime(dfFiltered["DateJoin"], format=('%Y-%m-%d %H:%M'))
    df2["DateJoin"] = pd.to_datetime(df2["DateJoin"], format=('%Y-%m-%d %H:%M'))

    # left join GA to tvdata
    dfFiltered.set_index('DateJoin', inplace=True)
    df2.set_index('DateJoin', inplace=True)
    joineddata = df2.join(dfFiltered, how='left')

    # rolling sum to get pageviews before and after ad
    joineddata["beforeAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(1)
    joineddata["afterAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(-attributionwindow+1)

    # only include rows that are a match
    joineddata = joineddata[joineddata['country'].str.len() > 0]
    joineddata.loc[joineddata['beforeAd'] > 0, 'prop'] = joineddata["afterAd"]/joineddata["beforeAd"] - 1
    joineddata.loc[joineddata['beforeAd'] == 0, 'prop'] = 1
    joineddata.loc[joineddata['afterAd'] == 0, 'prop'] = 0

    results = pg.ttest(joineddata["prop"], 0.0, paired=False, alternative='greater', confidence=0.90)
    st.write(results)

    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    # pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    # power = float(round(results['power']*100, 2).to_string(index=False))
    # confidenceinterval = round((results['CI90%'].values[0][0]*100),2)

    pvalue = float(results['p-val']*100)
    power = float(results['power']*100)
    confidenceinterval = (results['CI90%'].values[0][0]*100)


    spend = joineddata["amount"].sum()
    numberofads = len(joineddata["amount"])

    if confidenceinterval > 0:
        # CPL = avgspend / avgchange
        CPL = (spend / numberofads) / confidenceinterval
    else:
        # CPL = 0
        CPL = 0


    return medium, dimension, i, confidenceinterval, pvalue, power, spend, numberofads, CPL

# Plot charts of various metrics - for Deep Dive page
def PlottingDeepDive(table, metric):
    # plot bar chart of average change vs whatever metric
    # dont show average change duplicated in hover, show p value, conf interval and % change
    fig = px.bar(table, x=metric, y='Average Change (%)',
                 hover_data={'Average Change (%)': False, 'P-value (%)': True, 'Confidence Interval (%)': True},
                 labels={metric: ' ', 'Average Change (%)': ' '}, text="% change")
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name'] = 'Uplift'

    # change layout to hide axes, initialise second y axis
    fig.update_layout(xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=False),
                      yaxis2=dict(
                          title="yaxis2 title",
                          titlefont=dict(
                              color="#ff7f0e"
                          ),
                          tickfont=dict(
                              color="#ff7f0e"
                          ),
                          anchor="free",
                          overlaying="y",
                          side="left",
                          position=0.15
                      ))
    # formatting changes, title etc
    fig.update_xaxes(title_font=dict(size=16, family='Source Sans Pro', color='white'))
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(
        title={
            'text': "Uplift by " + metric,
            'y': 1,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},yaxis_tickformat='$')
    # change colour based on significance
    fig.update_traces(marker_color=table["Colour"])
    fig.update_xaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white',type='category')
    fig.update_yaxes(zeroline=True, zerolinewidth=0.01, zerolinecolor='white')

    # make negative CPL zero
    table["Cost Per Unit Lift ($)"] = np.where(table["Cost Per Unit Lift ($)"] < 0, 0, table["Cost Per Unit Lift ($)"])
    # if its 0, make it blank, otherwise format as currency
    table["Cost Per Unit Lift"] = np.where(table["Cost Per Unit Lift ($)"] == 0, "", "$" + (table["Cost Per Unit Lift ($)"]).map('{:,.0f}'.format))
    # change text colour based on significance - white is sig, grey is not
    table["TextColour"] = np.where(table["P-value (%)"] < 10, 'white', '#5A5A5A')
    # add line for CPL on 2nd y axis with lines, markers and text, and with significance based colours
    fig.add_trace(go.Scatter(x=table[metric], y=table["Cost Per Unit Lift ($)"],yaxis="y2",mode="markers+text",name="Cost Per Unit Lift ($)",textfont_color=table["TextColour"],marker_color=table["TextColour"],line_color="white",text=table["Cost Per Unit Lift"],textposition="top center"))
    # plot chart
    st.plotly_chart(fig, use_container_width=True)



### ------------------------- OPTIMISATION PAGE -----------------------------------------------



# Plot Response Curves for Uplift vs Ads per week - for Optimisation page
def ResponseCurves(df1, df2, attributionwindow, form):

    # find list of mediums
    medium = df1["medium"].unique()

    # set df1Filtered as df1 (no need just dont wanna rename rest of function)
    df1Filtered = df1

    # make sure dates are overlapping
    df2Filtered = df2[df2["DateJoin"] <= df1Filtered["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    df2Filtered = df2[df2["DateJoin"] >= df1Filtered["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]

    # create empty dataframe for table
    # table = pd.DataFrame({'Channel': pd.Series(dtype='str'),
    #
    #                 'Average Change (%)': pd.Series(dtype='str'),
    #                 'P-value (%)': pd.Series(dtype='str'),
    #                 'Confidence Interval (%)': pd.Series(dtype='str'),
    #                 'Spend ($)': pd.Series(dtype='float'),
    #                 '# of Ads': pd.Series(dtype='float'),
    #                 'Avg Cost Per Ad ($)': pd.Series(dtype='float'),
    #                 'Cost Per Unit Lift ($)': pd.Series(dtype='float')})

    # initialise empty curves dictionary and empty dataframe
    df_dict = {}
    totalcurve_dict = {}
    df = pd.DataFrame()
    # loop through each channel and run LiftCurves function
    for i in medium:

        # make sure there's more than one ad, otherwise throws error
        if i == "ABU DHABI SPORT 1" or i == "KSA SPORTS 1":
            continue
        if len(df1Filtered[df1Filtered["medium"] == i].index) > 1:

            # run LiftCurves function and store in variables
            # pvalue, power, confidenceinterval, avgchange, spend, numberofads, df1, curve_dict = LiftCurves(df1Filtered, df2Filtered, i, attributionwindow, form)
            df1, curve_dict = LiftCurves(df1Filtered, df2Filtered, i, attributionwindow, form)

            # append curvedict from function to totalcurve dictionary
            totalcurve_dict.update(curve_dict)
            # append df1 from function to df
            df = df.append(df1)
            # get channel, alpha and beta list from the df and zip them up into alphabeta
            channel_list = list(df['medium'])
            alpha_list = list(df['a'])
            beta_list = list(df['b'])
            alpha_beta = list(zip(alpha_list,beta_list))

            # loop through and construct dictionary with parameters
            for idx, key in enumerate(channel_list):
                df_dict[key] = alpha_beta[idx]

            # for each medium, append positive uplift and cost to curves dictionary
            #curves[i] = [prop2,amountPerAd]
            # calculate cost per lift
            # if avgchange > 0:
            #     CPL = (spend/numberofads)/avgchange
            # else:
            #     CPL = 0
            # append to table
            # table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, numberofads, spend/numberofads, CPL]

    # sort table by average change
    # table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    # reset index
    # table = table.reset_index(drop=True)

    return df_dict, df, totalcurve_dict

# Lift function as above but also with the curves being plot - for Optimiser page
def LiftCurves(df1, df2, medium, attributionwindow, form):

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1["medium"] == medium]
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)

    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= df2["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= df2["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]

    # set DateJoin to datetime
    dfFiltered["DateJoin"] = pd.to_datetime(dfFiltered["DateJoin"], format=('%Y-%m-%d %H:%M'))
    df2["DateJoin"] = pd.to_datetime(df2["DateJoin"], format=('%Y-%m-%d %H:%M'))

    # left join GA to tvdata
    dfFiltered.set_index('DateJoin', inplace=True)
    df2.set_index('DateJoin', inplace=True)
    joineddata = df2.join(dfFiltered, how='left')

    # rolling sum to get sum before and after ad
    joineddata["beforeAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(1)
    joineddata["afterAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(-attributionwindow+1)

    # only include rows that are a match
    joineddata = joineddata[joineddata['country'].str.len() > 0]
    joineddata.loc[joineddata['beforeAd'] > 0, 'prop'] = joineddata["afterAd"]/joineddata["beforeAd"] - 1
    joineddata.loc[joineddata['beforeAd'] == 0, 'prop'] = 1
    joineddata.loc[joineddata['afterAd'] == 0, 'prop'] = 0

    # run the t-test
    # results = pg.ttest(joineddata["prop"], 0.0, paired=False, alternative='greater', confidence=0.90)
    #
    # # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    # pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    # power = float(round(results['power']*100, 2).to_string(index=False))
    # confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    # avgchange = round(np.median(joineddata["prop"])*100,2)
    #avgchange = round(((sum(joineddata["afterAd"]) / sum(joineddata["beforeAd"]) - 1) * 100), 2)

    # get total spend and number of ads
    # spend = joineddata["amount"].sum()
    # numberofads = len(joineddata["amount"])

    # reset index, filter to only include columns needed and then only include ads that had a positive uplift
    joineddata2 = joineddata.reset_index(drop=False)
    joineddata2 = joineddata2[["medium","DateJoin", "beforeAd", "afterAd"]]
    joineddata2 = joineddata2[joineddata2["afterAd"] > joineddata2["beforeAd"]]

    # create chart variable that has ad uplift for each ad, by date and by channel
    chart = pd.DataFrame({'Ad Uplift': joineddata2["afterAd"]-joineddata2["beforeAd"], 'DateJoin': joineddata2["DateJoin"], 'Medium': medium})

    # create week variable
    chart['Week'] = pd.to_datetime(chart['DateJoin'], format='%Y-%m-%d %H:%M:%S')
    # convert to ISO week number
    chart["Week"] = chart["Week"].dt.isocalendar().week
    # group by channel and week and sum total Ad Uplift and # of Ads
    chart = chart.groupby(['Medium', 'Week']).agg({'Ad Uplift': 'sum', 'DateJoin': 'count'}).reset_index().rename(columns={'DateJoin' : 'Ads'})
    # reset index
    chart = chart.reset_index(drop=False)

    # Logarithmic response curve
    def logarithmic(x, k, c):
        return (k+c) * np.log(x + 1)

    # Square root response curve
    def sqrt(x, k, c):
        return k * np.sqrt(c * x)

    # # Power response curve
    # def power(x, k, a):
    #     return k * np.power(x, a)
    #
    # # Arctangent response curve
    # def atan(x, k, c, d):
    #     return k * np.arctan(c * x)


    functions = {
    'Logarithmic': logarithmic,
    'Square Root': sqrt
    }

    # sort by Ads, cast ads and ad uplift as float
    chart = chart.sort_values(by="Ads")
    chart["Ads"] = chart["Ads"].astype(float)
    chart["Ad Uplift"] = chart["Ad Uplift"].astype(float)

    # set xdata and ydata
    xdata = chart['Ads']
    ydata = chart['Ad Uplift']

    # run curve_fit function to get params in popt
    popt, pcov = curve_fit(functions[form], xdata,ydata, maxfev=10000)

    # Generate new x values for the curve
    x_curve = np.arange(0,max(xdata)*1.3,1)

    # Evaluate the curve at the new x values using the fitted parameters
    y_curve = functions[form](x_curve, popt[0], popt[1])

    # make df1 with medium, a and b for parameters
    df1 = pd.DataFrame({'medium': medium, 'a': [popt[0]], 'b': [popt[1]] })

    # make empty dictionary to hold medium as its key, a and b as parameters, xdata and ydata, xcurve and ycurve
    curve_dict = {}
    curve_dict[medium] = {'a': [popt[0]], 'b': [popt[1]], 'xdata' : xdata.tolist(), 'ydata' : ydata.tolist(), 'x_curve' : x_curve.tolist(), 'y_curve' : y_curve.tolist() }

    # return pvalue, power, confidenceinterval, avgchange, spend, numberofads, df1, curve_dict
    return df1, curve_dict

# Optimiser Function - for Optimisation page
def OptimiseBudgets(budget, df_dict, df, form):

    # Logarithmic response curve
    def logarithmic(x, k, c):
        return (k+c) * np.log(x + 1)

    # Square root response curve
    def sqrt(x, k, c):
        return k * np.sqrt(c * x)

    # define functions
    functions = {
        'Logarithmic': logarithmic,
        'Square Root': sqrt
    }

    # set budget to budget
    budget = budget

    # define channel optimiser function
    def channel_optimizer(int_budget_list, channels):
        res_list = [functions[form](int_budget_list[i], df_dict[channels[i]][0], df_dict[channels[i]][1] ) for i in
                    range(len(int_budget_list))]
        calculation = (sum(res_list))

        return -1 * calculation

    # define intitial budget where its spread equally
    def int_budget_maker(number_channels, budget):
        '''equaly weughted budget maker'''
        budget_share = budget / number_channels
        initial_budget = [budget_share for i in range(number_channels)]
        return initial_budget

    # define constraints
    def equal_cons(x):
        '''must be a constraint equal to the budget'''
        x_list = []
        for i in range(len(x)):
            x_list.append(x[i])

        return sum(x_list) - budget

    # define bounds
    bounds = []
    for x in range(len(df)):
        bounds.append((0, budget))

    # define constraints
    constraint = {'type': 'eq'
        , 'fun': equal_cons
                  # ,'args': (len(to_budget_df),)
                  }
    constraint = [constraint]

    # call minimize function with all the above to get results
    result = minimize(channel_optimizer, int_budget_maker(len(df), budget), args=(list(df['medium'].unique()))
             , jac='3-point', hess=None, bounds=bounds, constraints=constraint)

    # make df_results dataframe with channel names and optimal ads per week
    df_results = pd.DataFrame(list(zip(df_dict.keys(), result.x)), columns=['Channel', 'Optimal Ads Per Week'])
    # round optimal ads per week and no decimal places
    df_results['Optimal Ads Per Week'] = df_results['Optimal Ads Per Week'].round(decimals=0)
    # sort by highest optimal ads
    df_results = df_results.sort_values(by="Optimal Ads Per Week",ascending=False).reset_index(drop=True)
    # output as streamlit dataframe
    # st.dataframe(df_results.style.format(
    #     subset=["Optimal Ads Per Week"], formatter="{:.0f}"), use_container_width=True)
    return df_results


def ParallelMerge(df_results):
    x = ["Spot Length", "Ad Language", "TV Program Name", "Spot Hour"]
    table_list = Parallel(n_jobs=1)(
        delayed(MergingTables)(df_results, x[j]) for j in
        range(len(x)))

    st.write(table_list)


def MergingTables(df_results, i):
    table = st.session_state[i]
    table = table[table["P-value (%)"] < 10]
    table = table.sort_values(['Medium', 'Confidence Interval (%)'],
                              ascending=[True, False])
    table = table.groupby('Medium').first()
    table = table.reset_index()
    table = table[["Medium", "Dimension"]]
    table = table.rename({'Dimension': i, 'Medium': 'Channel'}, axis='columns')
    df_results = pd.merge(df_results, table, on='Channel', how='left')

    return df_results


### ------------------------- EXTRA -----------------------------------------------


def LiftDeepDive2(df1, df2, medium, attributionwindow):
    start = time.time()

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1["medium"] == medium]
    #end = time.time()
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)
    #end2 = time.time()

    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)
    #end3 = time.time()
    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= df2["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= df2["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]
    #end4 = time.time()
    dfFiltered["DateJoin"] = pd.to_datetime(dfFiltered["DateJoin"], format=('%Y-%m-%d %H:%M'))
    df2["DateJoin"] = pd.to_datetime(df2["DateJoin"], format=('%Y-%m-%d %H:%M'))
    #end5 = time.time()


    # left join GA to tvdata

    # joineddata = df2.merge(dfFiltered, on="DateJoin",how='left')
    dfFiltered.set_index('DateJoin', inplace=True)
    df2.set_index('DateJoin', inplace=True)
    joineddata = df2.join(dfFiltered, how='left')

    #end6 = time.time()
    #joineddata = joineddata.drop_duplicates(subset=['DateJoin'])

    joineddata["beforeAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(1)



    joineddata["afterAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(-attributionwindow+1)
    #end7 = time.time()

    # only include rows that are a match
    joineddata = joineddata[joineddata['country'].str.len() > 0]
    joineddata.loc[joineddata['beforeAd'] > 0, 'prop'] = joineddata["afterAd"]/joineddata["beforeAd"] - 1
    joineddata.loc[joineddata['beforeAd'] == 0, 'prop'] = 1
    joineddata.loc[joineddata['afterAd'] == 0, 'prop'] = 0
    #end8 = time.time()

    results = pg.ttest(joineddata["prop"], 0.0, paired=False, alternative='greater', confidence=0.90)


    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    power = float(round(results['power']*100, 2).to_string(index=False))
    confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    avgchange = round(np.median(joineddata["prop"])*100,2)
    #avgchange = round(((sum(joineddata["afterAd"]) / sum(joineddata["beforeAd"]) - 1) * 100), 2)

    spend = joineddata["amount"].sum()
    numberofads = len(joineddata["amount"])


    return pvalue, power, confidenceinterval, avgchange, spend, numberofads

# CalculateLift but for dimension - tweaked for StatsTable - for Deep Dive page
def CalculateLift2(df1, df2, medium, attributionwindow, i, dimension):

    # create new copy of tv data by filtering for medium and then dimension selected
    dfFirstFilter = df1[df1["medium"] == medium]
    dfFiltered = dfFirstFilter[dfFirstFilter[dimension] == i]


    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)
    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # find max and min date of kpi data
    maxdateKPI = df2["DateJoin"].max()
    mindateKPI = df2["DateJoin"].min()

    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= maxdateKPI + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= mindateKPI - pd.Timedelta(minutes=attributionwindow)]


    # initialise empty lists to output to
    beforeAd = []
    afterAd = []
    amountPerAd = []
    prop = []
    prop2 = []
    count = 0
    spend = dfFiltered['amount'].sum()

    # loop through each ad, merge the two dfs on date, then calculate difference between previous X rows and current + X-1 rows
    for i in dfFiltered.index:

        # x is the date of the ad i
        x = dfFiltered["DateJoin"][i]
        # find the row number of the matching date
        y = df2[df2['DateJoin'] == x].index.values
        # make y integer
        y = int(y)
        # make copy of new df of just pageviews
        df3 = df2["Pageviews"]

        # append sum of before ad for x minutes and after ad to empty lists
        beforeAd.append(df3[y-attributionwindow:y].sum())
        afterAd.append(df3[y:y+attributionwindow].sum())
        # calculate the % uplift for each ad then append each to prop empty list and run t-test on this versus zero
        prop.append((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1)
        # add 1 to count to keep number of ads
        count +=1


    # format average % uplift as string
    avgchange = round(((sum(afterAd) / sum(beforeAd) - 1) * 100), 2)

    # run t-test and store as results
    results = pg.ttest(prop, 0.0, paired=False, alternative='greater', confidence=0.90)
    # print(results)
    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    power = float(round(results['power']*100, 2).to_string(index=False))
    confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    avgspend = float(spend)/float(count)

    # return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAdComp, afterAdComp, beforeAdCompChart, afterAdCompChart, charttest
    return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAd, afterAd, spend, count, avgspend

# Table of channels with their stats - for By Channel page
def ChannelStatsTable(df1, df2, attributionwindow):

    # find list of mediums
    medium = df1["medium"].unique()
    # create empty table
    table = pd.DataFrame({'Channel': pd.Series(dtype='str'),
                    'Average Change (%)': pd.Series(dtype='str'),
                    'P-value (%)': pd.Series(dtype='str'),
                    'Confidence Interval (%)': pd.Series(dtype='str'),
                    'Spend ($)': pd.Series(dtype='float'),
                    '# of Ads': pd.Series(dtype='float'),
                    'Avg Cost Per Ad ($)': pd.Series(dtype='float'),
                    'Cost Per Unit Lift ($)': pd.Series(dtype='float')})

    # initialise empty curves dictionary
    curves = {}
    # loop through each channel and run CalculateLift function
    for i in medium:
        # print(i)
        # print("channelstatstable")
        # make sure there's more than one ad, otherwise throws error
        if len(df1[df1["medium"] == i].index) > 1:

            # run CalculateLift function and store in variables
            pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(df1, df2, i, attributionwindow)
            # for each medium, append positive uplift and cost to curves dictionary
            curves[i] = [prop2,amountPerAd]
            # calculate cost per lift
            CPL = avgspend / avgchange
            # append to table
            table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, count, avgspend, CPL]

    # sort table, reset index and return table and curves
    table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)
    return table, curves

# T-test function used in ChannelStatsTable - for By Channel page
def CalculateLift(df1, df2, medium, attributionwindow):

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1["medium"] == medium]

    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)
    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # find max and min date of kpi data
    maxdateKPI = df2["DateJoin"].max()
    mindateKPI = df2["DateJoin"].min()

    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= maxdateKPI + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= mindateKPI - pd.Timedelta(minutes=attributionwindow)]

    # initialise empty lists to output to
    beforeAd = []
    afterAd = []
    amountPerAd = []
    prop = []
    prop2 = []
    count = 0
    spend = dfFiltered['amount'].sum()
    date = []
    program = []
    language = []
    spotlength = []
    spothour = []
    spotposition = []

    # loop through each ad, merge the two dfs on date, then calculate difference between previous X rows and current + X-1 rows
    for i in dfFiltered.index:

        # x is the date of the ad i
        x = dfFiltered["DateJoin"][i]
        # find the row number of the matching date
        y = df2[df2['DateJoin'] == x].index.values
        # print(x,y)
        # print("POOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP")
        # make y integer
        if len(y) == 0:
            continue
        y = int(y)
        # make copy of new df of just pageviews
        df3 = df2["Pageviews"]


        # append sum of before ad for x minutes and after ad to empty lists
        beforeAd.append(df3[y-attributionwindow:y].sum())
        afterAd.append(df3[y:y+attributionwindow].sum())
        date.append(x)
        program.append(dfFiltered["tv_program_name"][i])
        language.append(dfFiltered["ad_language"][i])
        spotlength.append(dfFiltered["space"][i])
        spothour.append(dfFiltered["tv_spot_start_time_hour"][i])
        spotposition.append(dfFiltered["spot_position"][i])
        # print(dfFiltered["ad_language"][i])
        # print(language)
        # print("THIS IS BLAH")
        if df3[y-attributionwindow:y].sum() > 0:

            # calculate the % uplift for each ad then append each to prop empty list and run t-test on this versus zero
            prop.append((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1)
            # print(prop)
            #prop.append(np.log10(df3[y:y + attributionwindow].sum()) - np.log10(df3[y - attributionwindow:y].sum()))
            # if uplift was >0, append to prop2 - this is for response curves - only want to keep positive values
            if ((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1) > 0:
                prop2.append(((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1))
                #append the cost for each ad to list too
                amountPerAd.append(dfFiltered["amount"][i])
        else:
            prop.append(1)

        # add 1 to count to keep number of ads
        count +=1

    # format average % uplift as string
    # avgchange = round(((sum(afterAd) / sum(beforeAd) - 1) * 100), 2)
    avgchange = round(np.median(prop)*100,2)
    #print("this is prop")
    #print(medium)
    #print(prop)
    do = {'prop': prop, 'beforeAd': beforeAd, 'afterAd': afterAd}
    poop = pd.DataFrame(do)

    # if medium == "AL RAI TV":
    #     poop.to_csv(
    #         'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/POOOOOP.csv')
    #     prop.to_csv(
    #         'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/PROPafter.csv')
    # run t-test and store as results
    results = pg.ttest(prop, 0.0, paired=False, alternative='greater', confidence=0.90)
    # print(beforeAd)
    # print(afterAd)
    # print(medium)
    # print(np.seterr())

    # print(results)

    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    power = float(round(results['power']*100, 2).to_string(index=False))
    confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    avgspend = float(spend)/float(count)

    # return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAdComp, afterAdComp, beforeAdCompChart, afterAdCompChart, charttest
    return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd

# Table of dimensions with their stats ie Performance by Dimension - for Deep Dive page
def StatsTable2(df1, df2, medium, attributionwindow, dimension):

    # dfFiltered is df filtered for specific medium, then list of unique variables for that specific dimension ie if Spot Hour then 07,08,09 etc
    dfFiltered = df1[df1["medium"] == medium]
    var = dfFiltered[dimension].unique()

    # initialise table dataframe
    table = pd.DataFrame({'Dimension': pd.Series(dtype='str'),
                          'Average Change (%)': pd.Series(dtype='float'),
                          'P-value (%)': pd.Series(dtype='float'),
                          'Confidence Interval (%)': pd.Series(dtype='float'),
                          'Spend ($)': pd.Series(dtype='float'),
                          '# of Ads': pd.Series(dtype='float'),
                          'Avg Cost Per Ad ($)': pd.Series(dtype='float'),
                          'Cost Per Unit Lift ($)': pd.Series(dtype='float')}
                         )

    # loop through dimension and if it has more than one ad, run CalculateLift2 script
    for i in var:
        # print(i)
        if len(dfFiltered[dfFiltered[dimension] == i].index) > 1:
            pvalue, power, confidenceinterval, avgchange, spend, numberofads = LiftDeepDive3(df1, df2, medium, attributionwindow, i, dimension)

            if avgchange > 0:
                CPL = (spend/numberofads)/avgchange
            else:
                CPL = 0
            # append to table
            table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, numberofads, spend/numberofads, CPL]
    # sort table, reset index and return table
    table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)
    return table

def LiftDeepDive3(df1, df2, medium, attributionwindow, i, dimension):

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1[dimension] == i]
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)

    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # filter tv data to make sure its not longer or shorter than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= df2["DateJoin"].max() + pd.Timedelta(minutes=attributionwindow)]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= df2["DateJoin"].min() - pd.Timedelta(minutes=attributionwindow)]

    dfFiltered["DateJoin"] = pd.to_datetime(dfFiltered["DateJoin"], format=('%Y-%m-%d %H:%M'))
    df2["DateJoin"] = pd.to_datetime(df2["DateJoin"], format=('%Y-%m-%d %H:%M'))

    # left join GA to tvdata
    dfFiltered.set_index('DateJoin', inplace=True)
    df2.set_index('DateJoin', inplace=True)
    joineddata = df2.join(dfFiltered, how='left')

    # rolling sum to get pageviews before and after ad
    joineddata["beforeAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(1)
    joineddata["afterAd"] = joineddata["Pageviews"].rolling(attributionwindow).sum().shift(-attributionwindow+1)

    # only include rows that are a match
    joineddata = joineddata[joineddata['country'].str.len() > 0]
    joineddata.loc[joineddata['beforeAd'] > 0, 'prop'] = joineddata["afterAd"]/joineddata["beforeAd"] - 1
    joineddata.loc[joineddata['beforeAd'] == 0, 'prop'] = 1
    joineddata.loc[joineddata['afterAd'] == 0, 'prop'] = 0

    results = pg.ttest(joineddata["prop"], 0.0, paired=False, alternative='greater', confidence=0.90)


    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
    power = float(round(results['power']*100, 2).to_string(index=False))
    confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    avgchange = round(np.median(joineddata["prop"])*100,2)
    #avgchange = round(((sum(joineddata["afterAd"]) / sum(joineddata["beforeAd"]) - 1) * 100), 2)

    spend = joineddata["amount"].sum()
    numberofads = len(joineddata["amount"])


    return pvalue, power, confidenceinterval, avgchange, spend, numberofads
