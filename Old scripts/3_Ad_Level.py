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
        st.slider(label="Set the attribution window", min_value=5, max_value=45, value=30, step=5, key="attributionwindowadlevel")
        # initialise dropdown for brand selection
        st.selectbox('What brand do you want to analyse?', listofvalues, key="mediumadlevel")

        # run function
        pvalue, power, confidenceinterval, avgchange, visitsBefore, visitsAfter, beforeAd, afterAd, date, program, language, spotlength, spothour, spotposition, spend, count, avgspend, prop2, amountPerAd = CalculateLift(
            df1, df2, st.session_state.mediumadlevel, st.session_state.attributionwindowadlevel)

        chart = pd.DataFrame(
            {'Before Ad': beforeAd, 'After Ad': afterAd, 'Date': date, 'Program': program, 'Language': language,
             'Spot Length': spotlength, 'Spot Hour': spothour, 'Spot Position': spotposition})
        # calculate uplift
        chart["Uplift %"] = round((((chart["After Ad"] / chart["Before Ad"]) - 1) * 100), 1)
        # get index
        chart = chart.reset_index()

        col = st.radio('Pick how you want to color by', ['Program', 'Date'])

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
