import pandas as pd
import pingouin as pg
import numpy as np
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import os
import altair_transform

pd.options.display.float_format = '{:,}'.format

# Process TVData file
def ProcessTVData(tvdata):

    # filter relevant columns
    tvdata = tvdata[["country", "c_trans_date", "tv_spot_start_time_hour", "tv_spot_start_time_minute", "brand", "tv_program_name", "ad_language", "medium", "space", "amount", "mm", "yy", "spot_position"]]

    # dictionary to map old hours to new hours
    mapping_dict = {1: '01', 2: '02', 3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19', 20: '20', 21: '21', 22: '22', 23: '23', 24: '00', 25: '01', 26: '02', 27: '03', 28: '04', 29: '05'}

    # pad minutes
    tvdata["tv_spot_start_time_minute"] = tvdata["tv_spot_start_time_minute"].astype(str).str.pad(2, side='left', fillchar='0')

    # loop through each ad, map to dictionary and create DateJoin as datetime
    for i in tvdata.index:

        # map to dictionary
        tvdata.at[i, "tv_spot_start_time_hour"] = mapping_dict[tvdata["tv_spot_start_time_hour"][i]]

        # make DateJoin column a datetime object and add one hour for SAUDI ONLY - CHANGE THIS LATER
        tvdata.at[i, "DateJoin"] = tvdata["c_trans_date"][i].strftime('%Y-%m-%d') + " " + tvdata["tv_spot_start_time_hour"][i] + ":" + tvdata["tv_spot_start_time_minute"][i]
        tvdata.at[i, "DateJoin"] = pd.to_datetime(tvdata["DateJoin"][i], format=('%Y-%m-%d %H:%M'))

        # match timezones
        gmtplus3 = ['ksa', 'iraq', 'jordan', 'pan arab', 'Pan Arab', 'KSA', 'JORDAN', 'IRAQ', 'PAN ARAB', 'Ksa', 'Jordan', 'Iraq']
        if tvdata.at[i,"country"] in gmtplus3:
            tvdata.at[i, "DateJoin"] = tvdata["DateJoin"][i] + pd.Timedelta(hours=1)
            # for the spot hour table to show the correct hour
            tvdata.at[i,"tv_spot_start_time_hour"] = int(tvdata["tv_spot_start_time_hour"][i]) + 1

        # add one day if after midnight
        if tvdata["tv_spot_start_time_hour"][i] in ("00", "01", "02", "03", "04", "05"):
            tvdata.at[i, "DateJoin"] = tvdata["DateJoin"][i] + pd.Timedelta(days=1)


    # return as df1
    df1 = tvdata
    df1 = df1[["country", "tv_spot_start_time_hour", "brand", "tv_program_name", "ad_language", "medium", "space", "amount", "mm", "yy", "spot_position", "DateJoin"]]

    return df1

# Process KPIData file
def ProcessKPIData(kpidata):

    # pad zeroes
    kpidata["Hour"] = kpidata["Hour"].astype(str).str.pad(2, side = 'left', fillchar = '0')
    kpidata["Minute"] = kpidata["Minute"].astype(str).str.pad(2, side = 'left', fillchar = '0')

    # clean GA data, get rid of thousands separators
    if type(kpidata["Pageviews"][0]) == str:
        kpidata["Pageviews"] = kpidata["Pageviews"].str.replace(",", "").astype(int)
    #kpidata["Sessions"] = kpidata["Sessions"].str.replace(",", "").astype(int)

    # change date to same format as Statex file and convert to datetime object
    kpidata["DateJoin"] = kpidata["Date"].astype(str).str[0:4] + "-" + kpidata["Date"].astype(str).str[4:6] + "-" + kpidata["Date"].astype(str).str[6:8] + " " + kpidata["Hour"].astype(str) + ":" + kpidata["Minute"].astype(str)
    kpidata["DateJoin"] = pd.to_datetime(kpidata["DateJoin"], format=('%Y-%m-%d %H:%M'))

    # create new dataframe
    fulldates = pd.DataFrame()
    # create full range of dates with minimum being earliest and max being latest date of kpidata
    fulldates["DateJoin"] = pd.date_range(start=min(kpidata["DateJoin"]), end=max(kpidata["DateJoin"]),freq='min')

    # sort dates and reset index
    fulldates = fulldates.sort_values(by="DateJoin")
    fulldates = fulldates.reset_index(drop=True)
    # join kpidata onto full date range dataframe
    fulldates = pd.merge(fulldates,kpidata,how='left')
    # only keep datejoin and pageviews columns
    fulldates = fulldates[["DateJoin","Pageviews","Month"]]
    # fill blank cells with 0
    fulldates = fulldates.fillna(0)


    # return as df2
    df2 = fulldates
    return df2

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
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= maxdateKPI]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= mindateKPI]

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
        print(x,y)
        print("POOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOP")
        # make y integer
        if len(y) == 0:
            continue
        y = int(y[0])
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
        # if df3[y-attributionwindow:y].sum() == 0:
        #     prop.append(1)
        # else:
        # calculate the % uplift for each ad then append each to prop empty list and run t-test on this versus zero
        prop.append((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1)
        #prop.append(np.log10(df3[y:y + attributionwindow].sum()) - np.log10(df3[y - attributionwindow:y].sum()))
        # if uplift was >0, append to prop2 - this is for response curves - only want to keep positive values
        if ((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1) > 0:
            prop2.append(((df3[y:y+attributionwindow].sum()/df3[y-attributionwindow:y].sum())-1))
            #append the cost for each ad to list too
            amountPerAd.append(dfFiltered["amount"][i])

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
        beforeAd2 = pd.DataFrame(beforeAd)
        afterAd2 = pd.DataFrame(afterAd)
        if medium == "MBC 2":
            beforeAd2.to_csv(
                'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/PROPbef.csv')
            afterAd2.to_csv(
                'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/PROPafter.csv')
        # run t-test and store as results
        results = pg.ttest(prop, 0.0, paired=False, alternative='greater', confidence=0.90)
        print(len(beforeAd))
        print(len(afterAd))
        print(medium)
        print(np.seterr())

        print(results)

        # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
        pvalue = float(round(results['p-val']*100, 2).to_string(index=False))
        power = float(round(results['power']*100, 2).to_string(index=False))
        confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
        avgspend = float(spend)/float(count)

        # return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAdComp, afterAdComp, beforeAdCompChart, afterAdCompChart, charttest
        return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd

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

# Table of dimensions with their stats ie Performance by Dimension - for Deep Dive page
def StatsTable(df1, df2, medium, attributionwindow, dimension):
    # list of whatever dimension
    dfFiltered = df1[df1["medium"] == medium]
    var = dfFiltered[dimension].unique()
    # print(var)

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
            pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, spend, count, avgspend = CalculateLift2(df1, df2, medium, attributionwindow,i,dimension)
            # calculate CPL
            CPL = avgspend/avgchange
            # append to table
            table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, count, avgspend, CPL]

    # sort table, reset index and return table
    table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)
    return table

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
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] <= maxdateKPI]
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] >= mindateKPI]


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

# Plot charts for benchmarking vs industry - for By Channel page
def BenchmarkingCharts(listofchannels, table, benchmarkdf):
    # define list of metrics we want to benchmark
    listofmetrics = ['Average Change (%)', 'Avg Cost Per Ad ($)', 'Cost Per Unit Lift ($)']
    # loop through this list
    for x in listofmetrics:
        # initialise the df for charting
        chartdata = pd.DataFrame(
            {'Channel': pd.Series(dtype='str'), x: pd.Series(dtype='str'), "Metric": pd.Series(dtype='str'),
             "Benchmark": pd.Series(dtype='str')})
        # loop through channels
        for i in listofchannels:
            # create new table that only looks at that channel
            table2 = table[table["Channel"] == i]
            # now filter benchmark data for only that channel
            benchmarkdf2 = benchmarkdf[benchmarkdf["Channel"] == i]
            # drop duplicates
            table2 = table2.drop_duplicates()
            # reset index
            table2 = table2.reset_index(drop=True)
            # drop duplicates
            benchmarkdf2 = benchmarkdf2.drop_duplicates()
            # reset index
            benchmarkdf2 = benchmarkdf2.reset_index(drop=True)
            # get metric
            metric = round(float(table2[x][0]), 2)
            # get corresponding benchmark
            benchmark = round(float(benchmarkdf2[x][0]), 2)
            # get the difference
            if x == "Average Change (%)":
                metricvsbenchmark = round(100 * (float(table2[x][0]) / float(benchmarkdf2[x][0]) - 1), 0)
            else:
                metricvsbenchmark = -1 * round(100 * (float(table2[x][0]) / float(benchmarkdf2[x][0]) - 1), 0)
            # append result for each channel to chartdata df to plot along with the metric and the benchmark
            chartdata.loc[len(chartdata.index)] = [i, metricvsbenchmark, metric, benchmark]

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

# Plot ResponseCurves - for By Channel page
def ResponseCurves(df1, df2, attributionwindow):
    # define df1 and df2
    # df1 = df1[
    #     ["country", "sector", "category", "product", "brand", "subbrand", "tv_program_name", "total_breaks_in_program",
    #      "spot_number", "total_spots_in_break", "ad_language", "medium", "space", "amount", "spot_position",
    #      "DateJoin"]]
    # df2 = df2

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1

    # get list of channels
    listofchannels = dfFiltered["medium"].unique()

    # initialise empty lists
    upliftAd = []
    count = []
    DateJoin = []
    medium = []

    # loop through each channel
    for m in listofchannels:
        # filter and create new df for each channel
        dfFiltered2 = dfFiltered[dfFiltered["medium"] == m]

        # sort
        dfFiltered2 = dfFiltered2.sort_values(by=["DateJoin"], ascending=True)
        # reset index so its ordered, drop so we lose index column
        dfFiltered2 = dfFiltered2.reset_index(drop=True)
        # sort and reset index for df2 (kpi data) also
        df2 = df2.sort_values(by=["DateJoin"], ascending=True)
        df2 = df2.reset_index(drop=True)

        # find max and min date of kpi data
        maxdateKPI = df2["DateJoin"].max()
        mindateKPI = df2["DateJoin"].min()
        # filter tv data to make sure its not longer or shorter than kpi data
        dfFiltered2 = dfFiltered2[dfFiltered2["DateJoin"] <= maxdateKPI]
        dfFiltered2 = dfFiltered2[dfFiltered2["DateJoin"] >= mindateKPI]

        # loop through each ad, merge the two dfs on date, then calculate difference between previous X rows and current + X-1 rows
        for i in dfFiltered2.index:

            # x is the date of the ad i
            x = dfFiltered2["DateJoin"][i]
            # find the row number of the matching date
            y = df2[df2['DateJoin'] == x].index.values
            # make y integer
            # print("this is liftcurves")
            # print(x,y)
            y = int(y)
            # make copy of new df of just pageviews
            df3 = df2["Pageviews"]

            # append to empty lists
            upliftAd.append(df3[y:y + attributionwindow].sum() - df3[y - attributionwindow:y].sum())
            DateJoin.append(x)
            medium.append(m)

            # add 1 to count to keep number of ads
            count.append(1)

    # initialise df with required info
    chart = pd.DataFrame({'Ad Uplift': upliftAd, 'DateJoin': DateJoin, 'Medium' : medium, 'Ads' : count})
    # create week variable
    chart['Week'] = pd.to_datetime(chart['DateJoin'], format='%Y-%m-%d %H:%M:%S')
    # convert to ISO week number
    chart["Week"] = chart["Week"].dt.isocalendar().week
    # group by channel and week and sum total Ad Uplift and # of Ads
    chart = chart.groupby(['Medium', 'Week'])['Ad Uplift','Ads'].sum()
    # reset index
    chart = chart.reset_index(drop=False)
    # create key to join on
    chart["Key"] = chart["Medium"] + "-" + chart["Week"].astype(str)
    # output to file chart
    chart.to_csv('C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/chart.csv')

    # new df from TV data
    df1Week = df1
    # create week variable as datetime
    df1Week['Week'] = pd.to_datetime(df1Week['DateJoin'], format='%Y-%m-%d %H:%M:%S')
    # convert week to ISO num
    df1Week["Week"] = df1Week["Week"].dt.isocalendar().week
    # group by channel and week and sum spend/amount
    df1Week = df1Week.groupby(['medium','Week'])['amount'].sum()
    # reset index
    df1Week = df1Week.reset_index(drop=False)
    # create key to join on
    df1Week["Key"] = df1Week["medium"] + "-" + df1Week["Week"].astype(str)
    # output to weekly file
    df1Week.to_csv('C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Weekly.csv')

    # merge chart with df1Week so to match spend with uplift
    merged = df1Week.merge(chart, how="left", on="Key")
    # output to merged file
    merged.to_csv('C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/merged.csv')

    for i in merged['medium'].unique():
        # merged2 is merged filtered for specific channel
        merged2 = merged[merged["medium"] == i]
        # only include those with positive ad uplift
        merged2 = merged2[merged2["Ad Uplift"] >= 0]
        # create a new row so it goes through origin - add origin as a point
        new_row = {'Ad Uplift': 0.001, 'amount': 1.1}
        # add the new row
        merged2 = merged2.append(new_row, ignore_index=True)
        # sort by ad uplift
        merged2 = merged2.sort_values(by="Ad Uplift")
        # log spend
        merged2["amount"] = np.log(merged2["amount"])
        # output to merged2 file
        merged2.to_csv(
            'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/merged2.csv')
        # plot merged2, ad uplift against spend
        c = alt.Chart(merged2).mark_point().encode(
            x='amount',
            y='Ad Uplift'
        )
        # add regression line
        final_plot_amount = c + c.transform_regression(as_=('amount', 'Ad Uplift'), on='amount', regression='Ad Uplift',
                                                method='log').mark_line()
        st.subheader(i)
        # plot
        st.altair_chart(final_plot_amount)

    for j in merged['medium'].unique():
        # merged2 is merged filtered for specific channel
        merged3 = merged[merged["medium"] == j]
        # only include those with positive ad uplift
        merged3 = merged3[merged3["Ad Uplift"] > 0]
        # create a new row so it goes through origin - add origin as a point
        new_row = {'Ad Uplift': 0.01, 'Ads': 0.01}
        # add the new row
        merged3 = merged3.append(new_row, ignore_index=True)
        # sort by ad uplift
        merged3 = merged3.sort_values(by="Ad Uplift")
        # output to merged3 file
        merged3.to_csv(
            'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/merged3.csv')
        # plot merged3, ad uplift against ads
        c = alt.Chart(merged3).mark_point().encode(
            x='Ads',
            y='Ad Uplift'
        )

        test_dict = {'Ads': [5],'Ad Uplift': [100]}


        test_point = pd.DataFrame(test_dict)

        d = alt.Chart(test_point).mark_circle(size=150).encode(
            x='Ads',
            y='Ad Uplift'
        )

        trendline = c.transform_regression(as_=('Ads', 'Ad Uplift'), on='Ads',
                                                regression='Ad Uplift', method='log').mark_line()

        params = alt.Chart(merged3).mark_point().encode(
                x='Ads',
                y='Ad Uplift'
            ).transform_regression(params=True, as_=('Ads', 'Ad Uplift'), on='Ads',
                       regression='Ad Uplift', method='log').mark_line()

        final_plot_ads = c + d + trendline

        # print('------THESE ARE PARAMS----')
        # print(params.to_dict())

        st.subheader(j)
        # plot
        st.altair_chart(final_plot_ads)

    return True

# Benchmark any runs into an excel file - for By Channel page
def Benchmarking(table, df1):
    # make a copy of the t test results stored in table as table2
    table2 = table
    # make new column for attribution window
    table2["Attribution Window"] = st.session_state.attributionwindowbychannel
    # make column for brand - the brand the TV data is for
    table2["Brand"] = df1["brand"][1]
    # table2 = table2[table2["P-value (%)"] < 10]

    # make hdr a variable that makes it output header if new file otherwise if file already exists then just output data without headers
    hdr = False if os.path.isfile(
        'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv') else True
    # export to benchmarks file - APPEND data
    table2.to_csv(
        'C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv',
        mode='a', header=hdr, index=False)

    # find list of channels for that specific brand
    listofchannels = table["Channel"].unique()
    # create list
    d = {'Channel': listofchannels}
    # create dataframe with list
    benchmarkdf = pd.DataFrame(data=d)

    # import benchmark excel file
    benchmarkexcel = pd.read_csv(
        "C:/Users/maahaque/OneDrive - Publicis Groupe/Documents/2019/September/Attribution/TV Regression/Benchmarks.csv")
    # create new df merging channel list from before with the actual benchmarks so as to get common list
    benchmarkdf = pd.merge(benchmarkdf, benchmarkexcel, on=['Channel'])

    # filter df for attribution window selected
    benchmarkdf = benchmarkdf[benchmarkdf["Attribution Window"] == st.session_state.attributionwindowbychannel]
    # drop duplicates
    benchmarkdf = benchmarkdf.drop_duplicates()
    # reset index
    benchmarkdf = benchmarkdf.reset_index(drop=True)
    # return file
    return benchmarkdf, listofchannels


def CalculateLiftWilcoxon(df1, df2, medium, attributionwindow):


    # define df1 and df2
    df1 = df1[["country", "sector", "category", "product", "brand", "subbrand", "tv_program_name", "total_breaks_in_program", "spot_number", "total_spots_in_break", "ad_language", "medium", "space", "amount", "spot_position", "DateJoin"]]
    df2 = df2

    # create new copy of tv data by filtering for medium selected
    dfFiltered = df1[df1["medium"] == medium]
    # sort
    dfFiltered = dfFiltered.sort_values(by=["DateJoin"], ascending=True)
    # reset index so its ordered, drop so we lose index column
    dfFiltered = dfFiltered.reset_index(drop=True)
    # sort and reset index for df2 (kpi data) also
    df2 = df2.sort_values(by=["DateJoin"], ascending=True)
    df2 = df2.reset_index(drop=True)

    # find max date of kpi data
    maxdateKPI = df2["DateJoin"].max()

    # filter tv data to make sure its not longer than kpi data
    dfFiltered = dfFiltered[dfFiltered["DateJoin"] < maxdateKPI]


    # initialise empty lists to output to
    beforeAd = []
    afterAd = []
    beforeAdByMinute = []
    afterAdByMinute = []
    prop = []
    count = 0


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

        # append list to empty list above with pageviews upto x minutes before and after ad - FOR CHARTING
        beforeAdByMinute.append(list(df3[y-attributionwindow:y]))
        afterAdByMinute.append(list(df3[y:y+attributionwindow]))
        # append sum of before ad for x minutes and after ad to empty lists
        beforeAd.append(df3[y-attributionwindow:y].sum())
        afterAd.append(df3[y:y+attributionwindow].sum())
        # calculate the % uplift for each ad then append each to prop empty list and run t-test on this versus zero
        prop.append((df3[y:y+attributionwindow].sum() - df3[y-attributionwindow:y].sum()))
        # prop.append(np.log10(df3[y:y + attributionwindow].sum()) - np.log10(df3[y - attributionwindow:y].sum()))

        # add 1 to count to keep number of ads
        count +=1

    # calculate average of each minute before and after
    beforeAdComp = pd.DataFrame(beforeAdByMinute).mean(axis=0)
    afterAdComp = pd.DataFrame(afterAdByMinute).mean(axis=0)

    # format average % uplift as string
    avgchange = round(((sum(afterAd) / sum(beforeAd) - 1) * 100), 2)
    # run t-test and store as results
    #results = pg.ttest(prop, 0.0, paired=False, alternative='greater', confidence=0.90)
    results = stats.wilcoxon(prop, alternative='greater')
    # print(results)

    # extract pvalue, confidence interval from t-test - ALL ALREADY CONVERTED TO PERCENTAGE
    pvalue = float(round(results.pvalue*100,2))
    #power = float(round(results['power']*100, 2).to_string(index=False))
    #confidenceinterval = round((results['CI90%'].values[0][0]*100),2)
    #return pvalue, power, confidenceinterval, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAdComp, afterAdComp, beforeAdCompChart, afterAdCompChart, charttest
    return pvalue, avgchange, int(sum(beforeAd)/count), int(sum(afterAd)/count), beforeAd, afterAd

def ChannelStatsTableWilcoxon(df1, df2, attributionwindow):
    #find list of mediums
    medium = df1["medium"].unique()
    #create empty table
    table = pd.DataFrame({'Channel': pd.Series(dtype='str'),
                   'Average Change (%)': pd.Series(dtype='str'),
                   'P-value (%)': pd.Series(dtype='str'),
                   'Confidence Interval (%)': pd.Series(dtype='str')})
    #loop through each channel and run CalculateLift function
    for i in medium:
        pvalue, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd = CalculateLiftWilcoxon(df1, df2, i, attributionwindow)
        #append to table
        table.loc[len(table.index)] = [i, avgchange, pvalue, pvalue]

    table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)
    return table

def ChannelStatsTableByMonth(df1, df2, month, attributionwindow):

# find list of mediums

    medium = df1["medium"].unique()

    df1Filtered = df1[df1["mm"] == month]
    df2Filtered = df2[df2["Month"] == month]
    print(df1Filtered.head())
    print(df2Filtered.head())
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
        if len(df1Filtered[df1Filtered["medium"] == i].index) > 1:

            # run CalculateLift function and store in variables
            pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(df1Filtered, df2Filtered, i, attributionwindow)
            # for each medium, append positive uplift and cost to curves dictionary
            curves[i] = [prop2,amountPerAd]
            # calculate cost per lift
            CPL = avgspend / avgchange
            # append to table
            table.loc[len(table.index)] = [i, avgchange, pvalue, confidenceinterval, spend, count, avgspend, CPL]

    # sort table, reset index and return table and curves
    table = table.sort_values(by="Average Change (%)", axis=0, ascending=False)
    table = table.reset_index(drop=True)
    return table


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