import webbrowser

from mainfunctions import *

import streamlit as st
import plotly.graph_objs as go
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from xlsxwriter import Workbook
import numpy as np

#set up error handling if no files uploaded or page refresh
if 'count' not in st.session_state:
    st.write("Please upload files on the File Upload page!")
else:
    if st.session_state["count"] == 0:
        st.write("Please upload files on the File Upload page!")

    #if file uploaded, then this loop
    if st.session_state["count"] >= 1:
        st.session_state.attributionwindowbychannel = st.session_state.attributionwindowbychannel

        #title aligned center
        st.markdown("<h2 style='text-align: center; color: white; font-size: Source Sans Pro;'>Optimisation</h2>", unsafe_allow_html=True)

        # initialise slider
        #st.slider(label="Set the attribution window", min_value=5, max_value=120, value=30, step=5,key="attributionwindowbychannel")
        budget = st.slider(label="Set budget for ads per week", min_value=50, max_value=5000, value=500, step=5, key="budget")


        # initialise df1 (TVDATA) and df2 (KPIDATA) from file upload page
        df1 = st.session_state["df1"]
        df2 = st.session_state["df2"]

        # selectbox to pick which functional form the optimiser uses
        form = st.selectbox("Select functional form",["Logarithmic","Square Root"])

        html_str = f"""
         <p style='text-align: right; color: grey; font-size: 12px; font-family: Source Sans Pro;'>Optimising for a {st.session_state.attributionwindowbychannel} minute window. For a different window, please rerun Deep Dive page for that value.</p>
         """

        st.markdown(html_str, unsafe_allow_html=True)

        # call response curves function and get back parameters needed to plot
        df_dict, df, totalcurve_dict = ResponseCurves(df1, df2, st.session_state.attributionwindowbychannel, form)

        # call OptimiseBudgets functions to get back optimal ads per week
        df_results = OptimiseBudgets(budget, df_dict, df, form)


        for i in ["Spot Length", "Ad Language", "TV Program Name", "Spot Hour"]:
            table = st.session_state[i]
            table = table[table["P-value (%)"] < 10]
            table = table.sort_values(['Medium', 'Confidence Interval (%)'],
                                      ascending=[True, False])
            table = table.groupby('Medium').first()
            table = table.reset_index()
            table = table[["Medium", "Dimension"]]
            table = table.rename({'Dimension': i, 'Medium' : 'Channel'}, axis='columns')
            df_results = pd.merge(df_results, table, on='Channel',how='left')

        st.dataframe(df_results.style.format(
            subset=["Optimal Ads Per Week", "Spot Length"], formatter="{:.0f}"), use_container_width=True, height=(len(df_results) + 1) * 35 + 3)


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

        df_xlsx = to_excel(df_results)

        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)
        with col12:
            st.download_button(label='📥 Export To Excel',
                               data=df_xlsx,
                               file_name='OptimalTable.xlsx')

        #sort the dictionary that plots the curves in the same order as the optimal ads per week so the table has same order as response curves
        key_order = df_results["Channel"].tolist()
        totalcurve_dict = {k: totalcurve_dict[k] for k in key_order if k in totalcurve_dict}

        # empty dictionary for the optimal Ys - sum of pagelifts
        optimaly = []

        with st.expander("Optimisation Curves", expanded=False):
        # loop through each channel's parameters
            for key, value in totalcurve_dict.items():

                # Logarithmic response curve
                def logarithmic(x, k, c):
                    return (k + c) * np.log(x + 1)


                def sqrt(x, k, c):
                    return k * np.sqrt(c * x)


                functions = {
                    'Logarithmic': logarithmic,
                    'Square Root': sqrt
                }

                # set medium to be the dictionary's key (which is the channel)
                medium = key

                # set xdata and ydata as series
                xdata = pd.Series(value['xdata'])
                ydata = pd.Series(value['ydata'])

                # x_curve = pd.Series(value['x_curve'])
                # y_curve = pd.Series(value['y_curve'])

                # create a Plotly figure
                fig = go.Figure()

                # Update the layout to include the title and axis labels
                fig.update_layout(
                    title=medium,
                    xaxis_title='Ads Per Week',
                    xaxis_title_standoff=40,
                    yaxis_title='Total Uplift'
                )

                # find optimal point from the optimiser
                optimal_x = float(df_results[df_results['Channel'] == key]['Optimal Ads Per Week'])

                # range of x values to plot curve for, 30% greater than optimal value
                x_curve = np.arange(0, max(xdata) * 1.3, 1)

                # if max of xdata is greater than optimal, make the max based on max xdata, else 10% greater than optimal
                if max(xdata) > optimal_x:
                    x_curve = np.arange(0, max(xdata) * 1.3, 1)
                else:
                    x_curve = np.arange(0, optimal_x * 1.1, 1)

                # plot y logarithmic/sqrt curve for these x values with parameters from the fit curve
                y_curve = functions[form](x_curve, float(value['a'][0]), float(value['b'][0]))

                # find the corresponding optimal y point for the optimal x to plot optimal point on curve
                optimal_y = float(np.interp(optimal_x, x_curve, y_curve))
                # append each y to the empty list
                optimaly.append(optimal_y)

                # Add the data to the figure - first is points, second is log curve
                fig.add_trace(go.Scatter(x=xdata, y=ydata, name='Data', mode='markers'))
                fig.add_trace(go.Scatter(x=x_curve, y=y_curve, name='Log Function Fit'))

                # Add the optimal point as a separate trace
                fig.add_trace(go.Scatter(x=[optimal_x], y=[optimal_y], name='Optimal Point', mode='markers', marker=dict(size=17)))

                # Calculate R squared
                y_mean = (np.mean(ydata))
                ss_res = np.sum((ydata - functions[form](xdata, float(value['a'][0]), float(value['b'][0]))) ** 2)
                ss_tot = np.sum((ydata - y_mean) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                fig.add_annotation(text="R <sup>2</sup> : " + str(round(r_squared*100,1)) + "%",
                                   xref="paper", yref="paper",
                                   x=1, y=1.2, showarrow=False)
                fig.add_annotation(text="Optimal Ads : " + '{:,}'.format(int(optimal_x)),
                                   xref="paper", yref="paper",
                                   x=1, y=1.14, showarrow=False)
                fig.add_annotation(text="Uplift Expected : " + '{:,}'.format(int(optimal_y)),
                                   xref="paper", yref="paper",
                                   x=1, y=1.08, showarrow=False)

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)


                # @st.cache_data
                # def convert_df(df):
                #     df["Attribution Window"] = st.session_state.attributionwindowbychannel
                #     return df.to_csv().encode('utf-8')
                #
                #
                # csv = convert_df(df_results)
                #
                # col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = st.columns(12)
                # with col12:
                #     st.download_button(
                #         "📥 Export To Excel",
                #         csv,
                #         "OptimalTable.csv",
                #         "text/csv",
                #         key='optimaltable'
                #     )

        #
        # import igraph
        # import plotly.graph_objects as go
        #
        # # horizontal
        # G = igraph.Graph(directed=True)
        # lov = df_results["Channel"].tolist()
        # lov.insert(0,"Total Budget")
        #
        # mediaplanslider = st.slider("How many channels do you want the media plan for?",min_value=2,max_value=(len(lov)-1),value=6)
        #
        # lov = lov[0:mediaplanslider+1]
        # st.write(lov)
        # st.write(("Eng","Arb")*mediaplanslider)
        # lov = lov + ["Eng","Arb"]*(mediaplanslider)
        # st.write(lov)
        # G.add_vertices(lov)
        # #G.add_vertices(["Total Budget", "MBC 2", "MBC 1", "DUBAI TV", "Eng", "Arb", "Eng","Arb", "Eng","Arb", "Movie", "News", "Comedy","Movie","News","Movie"])
        # #G.add_edges([(0,1),(0,2),(0,3),(1,4),(1,5),(2,6),(2,7),(3,8),(3,9),(4,10),(4,11),(4,12),(6,13),(6,14),(9,15)])
        # G.add_edges([(0, 1),(0, 2),(1, 1+mediaplanslider), (1, 2+mediaplanslider), (2, 3+mediaplanslider),(2, 4+mediaplanslider)])
        #
        # # Get the layout of the graph
        # lay = G.layout('rt')
        #
        # # Create a dictionary of node positions
        # position = {k: (lay[k][1], -lay[k][0]) for k in range(len(G.vs))}
        #
        # # Define the X coordinates of the nodes
        # Xn = [position[k][0] for k in range(len(G.vs))]
        #
        # # Define the Y coordinates of the nodes
        # Yn = [-position[k][1] for k in range(len(G.vs))]
        #
        # # Define the node labels
        # labels = G.vs["name"]
        #
        # # Create the plot
        # fig = go.Figure()
        #
        # # Add the node markers
        # fig.add_trace(go.Scatter(x=Xn,
        #                          y=Yn,
        #                          mode='markers',
        #                          name='Nodes',
        #                          marker=dict(symbol='circle-dot',
        #                                      size=60,
        #                                      color='#6175c1',
        #                                      line=dict(color='rgb(50,50,50)', width=1)),
        #                          text=labels,
        #                          hoverinfo='text',
        #                          opacity=0.8))
        #
        # # Add the edge lines
        # for edge in G.get_edgelist():
        #     x0, y0 = position[edge[0]]
        #     x1, y1 = position[edge[1]]
        #     fig.add_trace(go.Scatter(x=[x0, x1, None],
        #                              y=[-y0, -y1, None],
        #                              mode='lines',
        #                              line=dict(width=1, color='grey'),
        #                              hoverinfo='none'))
        #
        #
        # # Remove the axis labels and ticks
        # fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        # fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        # annotations = []
        # for i in range(len(labels)):
        #     # Add the main annotation below the node
        #     annotations.append(dict(x=Xn[i], y=Yn[i], text=labels[i], showarrow=False,
        #                             font=dict(color='white', size=12)))
        #     # Add a second annotation above the node
        #     annotations.append(dict(x=Xn[i], y=Yn[i] + 0.3, text="Second annotation", showarrow=False,
        #                             font=dict(color='gray', size=10)))
        #
        # # Remove the plot background and gridlines
        # fig.update_layout(height=1000,plot_bgcolor='rgba(0,0,0,0)', hovermode='closest', showlegend=False,
        #                   margin=dict(b=20, l=5, r=5, t=40),
        #                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        #                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        #                   annotations=annotations)
        #
        # # Set the aspect ratio and size of the figure
        # st.plotly_chart(fig, use_container_width=True)
        #
