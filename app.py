import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
np.bool = np.bool_
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('df_final_VA.csv')

# Preprocess data for PCA and Clustering
df = df[df['Province'].notna()]
df_filtered = df[['Province', 'Year', 'Avg_Housing_Value', 'Total_Workers_Count', 'Avg_Total_CPI']]

# Create categories for Avg_Value (Low, Medium, High)
def categorize_avg_value(value):
    if value < df['Avg_Housing_Value'].quantile(0.33):
        return 'Low'
    elif value < df['Avg_Housing_Value'].quantile(0.66):
        return 'Medium'
    else:
        return 'High'

# Apply the categorization
df['Avg_Value_Class'] = df['Avg_Housing_Value'].apply(categorize_avg_value)

# Get the min and max year from the data
min_year = df['Year'].min()
max_year = df['Year'].max()


# Aggregate the data by province and year
df_grouped = df_filtered.groupby(['Province', 'Year']).agg({
    'Avg_Housing_Value': 'mean',
    'Total_Workers_Count': 'mean',
    'Avg_Total_CPI': 'mean',
}).reset_index()

# Standardize the features
features = ['Avg_Housing_Value', 'Total_Workers_Count', 'Avg_Total_CPI']
scaler = StandardScaler()
df_grouped[features] = scaler.fit_transform(df_grouped[features])

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_grouped[features])
df_grouped['PCA1'] = pca_components[:, 0]
df_grouped['PCA2'] = pca_components[:, 1]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=4)  # Adjust the number of clusters based on your data
df_grouped['Cluster'] = kmeans.fit_predict(df_grouped[['PCA1', 'PCA2']])

# Calculate the means of the features for each cluster
cluster_means = df_grouped.groupby('Cluster')[features].mean().reset_index()

# Dynamically create descriptions based on cluster feature means
cluster_descriptions = {}
for idx, row in cluster_means.iterrows():
    avg_value = row['Avg_Housing_Value']
    workers_count = row['Total_Workers_Count']
    cpi = row['Avg_Total_CPI']

    if avg_value > 0.5 and workers_count < 0:
        cluster_descriptions[row['Cluster']] = f"Cluster {row['Cluster']}: High Housing Prices, Low Workers"
    elif avg_value < 0 and workers_count > 0:
        cluster_descriptions[row['Cluster']] = f"Cluster {row['Cluster']}: Low Housing Prices, High Workers"
    elif cpi > 0.5:
        cluster_descriptions[row['Cluster']] = f"Cluster {row['Cluster']}: High CPI, Moderate Housing"
    else:
        cluster_descriptions[row['Cluster']] = f"Cluster {row['Cluster']}: Balanced Economic Factors"

# Group data for Time Series Forecasting
df_grouped_ts = df.groupby(['Province', 'Year'], as_index=False)['Avg_Housing_Value'].mean()

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
        html.H1(
        "Analyzing the Impact of Temporary Residents on Canada’s Housing Market",
        style={'textAlign': 'center', 'padding': '20px'}
    ),

    # Row for the first plot: Time Series Forecasting
    html.Div([
        html.H2("Forecasting Average Housing Price by Province"),
        html.Div([
            html.Div([
                html.Label("Select Province"),
                dcc.Dropdown(
                    id='province-dropdown-1',
                    options=[{'label': prov, 'value': prov} for prov in df_grouped_ts['Province'].unique()],
                    value='Ontario',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '25%', 'padding': '10px'}),

            html.Div([
                html.Label("Select Number of Years to Forecast"),
                dcc.Slider(
                    id='forecast-years-slider',
                    min=1,
                    max=10,
                    step=1,
                    value=5,  # default value
                    marks={i: str(i) for i in range(1, 11)},  # Creating marks for each year
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'display': 'inline-block', 'width': '75%', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        dcc.Graph(id="avg-value-time-series"),
    ], style={'padding': '20px'}),

    # Row for the second plot: Feature Relationship Scatter Plot
    html.Div([
        html.H2("Key Feature Relationships by Province"),
        html.Div([
            html.Div([
                html.Label("Select Province"),
                dcc.Dropdown(
                    id='province-dropdown-2',
                    options=[{'label': prov, 'value': prov} for prov in df['Province'].unique()],
                    value='Ontario',  
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

            html.Div([
                html.Label("Select Feature for X-axis"),
                dcc.Dropdown(
                    id='x-feature-dropdown',
                    options=[
                        {'label': 'Number of Study Permits Issued', 'value': 'Number_of_Study_Permits_Issued'},
                        {'label': 'Study Permits to Housing Price Ratio', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total Temporary Workers Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Temporary Workers To Housing Price Impact Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Combined Impact of Study Permits and Workers', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Average Total CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Bank Interest Rate (Weekly)', 'value': 'Avg_Bank_Interest_Rate_Weekly'},
                        {'label': 'Variable Mortgage Rate', 'value': 'Avg_Estimated_Variable_Mortgage_Rate'},
                        {'label': 'Effective Household Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Household_Interest_Rate'},
                        {'label': 'Effective Business Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Business_Interest_Rate'}
                    ],
                    value='Number_of_Study_Permits_Issued',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

            html.Div([
                html.Label("Select Feature for Y-axis"),
                dcc.Dropdown(
                    id='y-feature-dropdown',
                    options=[
                        {'label': 'Number of Study Permits Issued', 'value': 'Number_of_Study_Permits_Issued'},
                        {'label': 'Study Permits to Housing Price Ratio', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total Temporary Workers Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Temporary Workers To Housing Price Impact Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Combined Impact of Study Permits and Workers', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Average Total CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Bank Interest Rate (Weekly)', 'value': 'Avg_Bank_Interest_Rate_Weekly'},
                        {'label': 'Variable Mortgage Rate', 'value': 'Avg_Estimated_Variable_Mortgage_Rate'},
                        {'label': 'Effective Household Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Household_Interest_Rate'},
                        {'label': 'Effective Business Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Business_Interest_Rate'}
                    ],
                    value='Total_Workers_Count',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

            html.Div([
                html.Label("Select Feature for Circle Size"),
                dcc.Dropdown(
                    id='size-feature-dropdown',
                    options=[
                        {'label': 'Average Housing Price', 'value': 'Avg_Housing_Value'},
                        {'label': 'Study Permits to Housing Price Ratio', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total Temporary Workers Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Number of Class Titles', 'value': 'Number_of_Class_Titles'},
                        {'label': 'Temporary Workers To Housing Price Impact Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Combined Impact of Study Permits and Workers', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Average Total CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Variable Mortgage Rate', 'value': 'Avg_Estimated_Variable_Mortgage_Rate'}
                    ],
                    value='Avg_Housing_Value',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        html.Div([
            html.Label("Select Year Range"),
            dcc.RangeSlider(
                id='year-range-slider',
                min=df['Year'].min(),
                max=df['Year'].max(),
                step=1,
                marks={i: str(i) for i in range(df['Year'].min(), df['Year'].max() + 1, 1)},
                value=[df['Year'].min(), df['Year'].max()],
            ),
        ], style={'padding': '10px'}),

        dcc.Graph(id="feature-relationship-plot"),
    ], style={'padding': '20px'}),

    # Row for the third plot: PCA and KMeans Clustering (Scatter/Radar Toggle)
    html.Div([
        html.H2("Regional Housing Trend Clusters in Canada"),
        html.Div([
            html.Div([
                html.Label("Select Province"),
                dcc.Dropdown(
                    id='province-dropdown',
                    options=[{'label': prov, 'value': prov} for prov in df_grouped['Province'].unique()],
                    value='Ontario',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '25%', 'padding': '10px'}),

            html.Div([
                dcc.RadioItems(
                    id='chart-toggle',
                    options=[
                        {'label': 'Show Scatter Plot', 'value': 'scatter'},
                        {'label': 'Show Radar Chart', 'value': 'radar'}
                    ],
                    value='scatter',  # Default view
                    labelStyle={'display': 'inline-block', 'padding': '10px'}
                ),
            ], style={'display': 'inline-block', 'width': '75%', 'padding': '10px'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        dcc.Graph(id="regional-cluster-plot"),
    ], style={'padding': '20px'}),

        html.Div([
        html.H2("Impact of Occupation on Housing Demand in Canada", style={'textAlign': 'left', 'padding': '20px'}),

        # Container for dropdowns to be in line
        html.Div([
            html.Div([
                html.Label("Select Occupational Class Title", style={'textAlign': 'left', 'padding-right': '10px'}),
                dcc.Dropdown(
                    id='class-title-dropdown',
                    options=[{'label': title, 'value': title} for title in df['Most_Common_Class_Title'].unique()],
                    value=df['Most_Common_Class_Title'].unique()[0],
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Select Metric to Visualize", style={'textAlign': 'left', 'padding-right': '10px'}),
                dcc.Dropdown(
                    id='metric-dropdown',
                    options=[
                        {'label': 'Avg Housing Price', 'value': 'Avg_Housing_Value'},
                        {'label': 'CPI Index', 'value': 'CPI_Index'},
                        {'label': 'Combined Impact of Study Permits and Workers on Housing Prices', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Total Temporary Workers Count', 'value': 'Total_Workers_Count'},  # New metric
                        {'label': 'Number of Study Permits Issued', 'value': 'Number_of_Study_Permits_Issued'}  # New metric
                    ],
                    value='Avg_Housing_Value',
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '48%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Select Year", style={'textAlign': 'left', 'padding-right': '10px'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} for year in sorted(df['Year'].unique())],
                    value=df['Year'].min(),
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'padding': '10px', 'width': '48%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'justify-content': 'space-between'}),

        dcc.Graph(id="occupational-influence-plot", style={'padding': '20px'}),
    ], style={'padding': '20px'}),
        html.Div([
    html.H2("Trend Analysis of Selected Metrics"),
    html.Div([
        html.Div([
            html.Label("Select Province"),
            dcc.Dropdown(
                id='province-dropdown-trend',
                options=[{'label': prov, 'value': prov} for prov in df['Province'].unique()],
                value='Ontario',
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

        html.Div([
            html.Label("Select Metric"),
            dcc.Dropdown(
                id='metric-dropdown-trend',
                options=[
                        {'label': 'Average Housing Price', 'value': 'Avg_Housing_Value'},
                        {'label': 'Average Total CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Total Temporary Workers Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Study Permits to Housing Price Ratio', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Temporary Workers To Housing Price Impact Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Combined Impact of Study Permits and Workers', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Bank Interest Rate (Weekly)', 'value': 'Avg_Bank_Interest_Rate_Weekly'},
                        {'label': 'Variable Mortgage Rate', 'value': 'Avg_Estimated_Variable_Mortgage_Rate'},
                        {'label': 'Effective Household Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Household_Interest_Rate'},
                        {'label': 'Effective Business Interest Rate (Weekly)', 'value': 'Avg_Weekly_Effective_Business_Interest_Rate'}
                ],
                value='Avg_Housing_Value',
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

        html.Div([
            html.Label("Choose Detection Method"),
            dcc.Dropdown(
                id='method-dropdown',
                options=[
                    {'label': 'Z-Score Analysis', 'value': 'z_score'},
                    {'label': 'Interquartile Range (IQR) Analysis', 'value': 'iqr'}
                ],
                value='z_score',
                clearable=False,
                style={'width': '100%'}
            ),
        ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    dcc.Graph(id="trend-analysis-plot"),
], style={'padding': '20px'}),
])

# Callback for updating the graph based on selected province and forecast years
@app.callback(
    Output("avg-value-time-series", "figure"),
    [Input("province-dropdown-1", "value"),
     Input("forecast-years-slider", "value")]
)
def update_graph(selected_province, forecast_years):
    # Filter data based on selected province
    df_filtered = df_grouped[df_grouped['Province'] == selected_province]

    # Prepare data for ARIMA (keep 'Year' as a column and avoid setting it as index)
    df_filtered = df_filtered[['Year', 'Avg_Housing_Value']]

    # Fit an ARIMA model
    model = ARIMA(df_filtered['Avg_Housing_Value'], order=(5,1,0))  # ARIMA(p,d,q) with p=5, d=1, q=0
    model_fit = model.fit()

    # Forecast the future
    forecast = model_fit.forecast(steps=forecast_years)

    # Create future years for the forecast
    future_years = np.arange(df_filtered['Year'].max() + 1, df_filtered['Year'].max() + forecast_years + 1)

    # Combine the historical and forecasted data into one DataFrame
    forecasted_df = pd.DataFrame({
        'Year': future_years,
        'Avg_Housing_Value': forecast
    })

    # Plot the original data and forecast
    fig = px.line(df_filtered, x="Year", y="Avg_Housing_Value",
                  # title=f"Average Housing Price in {selected_province}",
                  labels={"Avg_Housing_Value": "Average Housing Price"},
                  markers=True)

    # Add the forecasted data as a continuous line with dotted style and markers (dots)
    fig.add_scatter(x=pd.concat([df_filtered['Year'], forecasted_df['Year']]),
                    y=pd.concat([df_filtered['Avg_Housing_Value'], forecasted_df['Avg_Housing_Value']]),
                    mode='lines+markers', name='Forecast Data',
                    line=dict(dash='dot', color='red', width=3),
                    marker=dict(symbol='circle', size=6, color='red'))  # Adding markers to forecasted data

    # Customize the layout and legend
    fig.update_layout(
        title=dict(x=0.5, xanchor='center'),  # Center the title
        xaxis_title="Year",
        yaxis_title="Average Housing Price",
        font=dict(family="Arial", size=12),
        hovermode="x unified",  # Display hover information for all traces on the x-axis
        template="plotly_white",  # Light background theme for better readability
        yaxis=dict(tickformat=".2f"),  # Format y-axis to two decimal places
        legend_title_text="Legend",  # Title for the legend
        legend=dict(
            x=1.05,  # Position the legend to the right outside the plot
            y=1,  # Position at the top
            borderwidth=2,  # Optional: Add a border around the legend
            font=dict(size=12),  # Font size of the legend
            bgcolor="rgba(255, 255, 255, 0.9)"  # Background color of the legend
        )
    )

    # Adding the legend and colors for both historical and forecasted data
    fig.add_scatter(x=df_filtered['Year'], y=df_filtered['Avg_Housing_Value'], mode='lines+markers', name='Historic Data',
                    line=dict(color='blue', width=3), marker=dict(symbol='circle', size=6, color='blue'))  # Historic data with blue dots

    return fig


# Callback for updating the plot based on selected features and year range
@app.callback(
    [
        Output("feature-relationship-plot", "figure"),
        Output('y-feature-dropdown', 'options')  # Update y-feature dropdown options
    ],
    [
        Input("province-dropdown-2", "value"),
        Input("x-feature-dropdown", "value"),
        Input("y-feature-dropdown", "value"),
        Input("size-feature-dropdown", "value"),
        Input("year-range-slider", "value")
    ]
)
def update_graph(selected_province, x_feature, y_feature, size_feature, year_range):
    # Filter data based on selected province and year range
    df_filtered = df[(df['Province'] == selected_province) &
                     (df['Year'] >= year_range[0]) &
                     (df['Year'] <= year_range[1])]

    # Update y-feature dropdown options dynamically
    available_y_options = [
        {'label': 'Number of Study Permits Issued', 'value': 'Number_of_Study_Permits_Issued'},
        {'label': 'Study_Permit_To_Value_Impact_Metric', 'value': 'Study_Permit_To_Value_Impact_Metric'},
        {'label': 'Total Workers Count', 'value': 'Total_Workers_Count'},
        {'label': 'Workers To Housing Price Impact Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
        {'label': 'Avg Total CPI', 'value': 'Avg_Total_CPI'},
        {'label': 'Avg CPI TRIM', 'value': 'Avg_CPI_TRIM'}
    ]
    # Remove x_feature from y_feature dropdown options
    available_y_options = [opt for opt in available_y_options if opt['value'] != x_feature]

    # Create a scatter plot with the selected size feature

    fig = px.scatter(df_filtered,
                 x=x_feature,
                 y=y_feature,
                 size=size_feature,  # Use selected size feature for the size of the dots
                #  title=f"Impact of {x_feature} vs. {y_feature} on Housing Prices in {selected_province} (Size Based on {size_feature})",
                 labels={x_feature: x_feature, y_feature: y_feature},
                 color="Avg_Value_Class",  # Color by the class of Avg_Value
                 hover_data=["Year", "Province", "Avg_Value_Class"])  # Display year, province, and class on hover


    # Update the layout to make the size scale more visible and format the legend
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        font=dict(family="Arial", size=12),
        hovermode="x unified",
        showlegend=True,  # Show legend
        legend=dict(
            title='Avg_Value Class',
            itemsizing='constant',  # Make legend items the same size
            bordercolor='black',  # Add a border to the legend
            borderwidth=1,
        )
    )

    return fig, available_y_options


# Callback for updating the scatter plot and radar chart for clustering
@app.callback(
    Output("regional-cluster-plot", "figure"),
    [
        Input("province-dropdown", "value"),
        Input("chart-toggle", "value")
    ]
)
def update_cluster_plot(selected_province, chart_type):
    # Filter data based on selected province
    df_province = df_grouped[df_grouped['Province'] == selected_province]

    if chart_type == 'scatter':
        # Convert 'Cluster' to a categorical (string) type to ensure discrete color scale
        df_province['Cluster'] = df_province['Cluster'].astype(str)

        # Add descriptive hover text
        df_province['hover_text'] = df_province.apply(lambda row: f"Cluster: {cluster_descriptions[int(row['Cluster'])]}<br>Avg Value: {row['Avg_Housing_Value']:.2f}<br>Workers Count: {row['Total_Workers_Count']:.2f}<br>CPI: {row['Avg_Total_CPI']:.2f}<br>Year: {row['Year']}", axis=1)

        # Create a scatter plot showing the clusters in 2D
        fig = px.scatter(df_province,
                        x='PCA1',
                        y='PCA2',
                        color='Cluster',  # Cluster as color
                        # title=f"Clustered Regional Housing Trends for {selected_province}",
                        labels={'PCA1': 'Economic Dimension 1', 'PCA2': 'Economic Dimension 2'},
                        hover_name='Province',
                        hover_data={'Year': True, 'Avg_Housing_Value': True, 'Total_Workers_Count': True, 'Avg_Total_CPI': True},
                        color_discrete_sequence=px.colors.qualitative.Set1,  # Ensure discrete color scale
                        category_orders={'Cluster': sorted(df_province['Cluster'].unique())})  # Ensure ordered cluster appearance

        # Set the cluster descriptions in the legend
        fig.for_each_trace(lambda t: t.update(name=cluster_descriptions[int(t.name)]))

        # Adjust point size based on cluster size and scale it to ensure clusters are larger
        cluster_sizes = df_province['Cluster'].value_counts().to_dict()
        df_province['point_size'] = df_province['Cluster'].map(cluster_sizes).apply(lambda x: x * 12)  # Scale by a larger factor for visibility

        # Update scatter plot with adjusted point sizes
        fig.update_traces(marker=dict(size=df_province['point_size'], opacity=0.7, line=dict(width=0)))

        # Add year labels as annotations without arrows
        annotations = []
        for i, row in df_province.iterrows():
            annotations.append(
                dict(
                    x=row['PCA1'],
                    y=row['PCA2'],
                    xref='x',
                    yref='y',
                    text=f"{row['Year']}",  # Display only the year text
                    showarrow=False,  # No arrow for cleaner look
                    font=dict(
                        size=12,  # Font size
                        color='black',  # Text color for readability
                        family='Arial'  # Font family
                    ),
                    # yshift=10  # Shift the label slightly above the point
                )
            )

        # Add annotations to the plot
        fig.update_layout(annotations=annotations)

        # Customize the layout to add the legend heading and boundary
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Economic Dimension 1 (PCA1)",
            yaxis_title="Economic Dimension 2 (PCA2)",
            showlegend=True,
            legend=dict(
                title="Cluster Descriptions",  # Add title to the legend
                bgcolor="rgba(255, 255, 255, 0.5)",  # Add background color for better visibility
                bordercolor="black",  # Add a border around the legend
                borderwidth=2  # Width of the border
            )
        )

    elif chart_type == 'radar':
        # Create a radar chart to show cluster characteristics
        cluster_means = df_province.groupby('Cluster')[features].mean().reset_index()
        fig = go.Figure()

        for cluster in cluster_means['Cluster']:
            fig.add_trace(go.Scatterpolar(
                r=cluster_means[cluster_means['Cluster'] == cluster][features].values.flatten(),
                theta=features,
                fill='toself',
                name=cluster_descriptions[cluster]  # Meaningful name for the cluster
            ))

        # Customize the layout for radar chart with legend heading and boundary
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-3, 3])
            ),
            title=f"Radar Chart of Cluster Characteristics for {selected_province}",
            showlegend=True,
            legend=dict(
                title="Cluster Descriptions",  # Add title to the legend
                bgcolor="rgba(255, 255, 255, 0.5)",  # Add background color for better visibility
                bordercolor="black",  # Add a border around the legend
                borderwidth=2  # Width of the border
            )
        )

    return fig


# Callback to update the bar chart based on dropdown selections
@app.callback(
    Output("occupational-influence-plot", "figure"),
    [Input("class-title-dropdown", "value"),
     Input("metric-dropdown-trend", "value"),
     Input("year-dropdown", "value")]
)
def update_occupational_influence_plot(selected_class_title, selected_metric, selected_year):
    # Filter the dataframe for the selected class title and year
    df_filtered_class_year = df[(df['Most_Common_Class_Title'] == selected_class_title) &
                                (df['Year'] == selected_year)]

    # Create a bar chart with the selected metric for each province
    fig = px.bar(
        df_filtered_class_year,
        x='Province',  # Group by Province
        y=selected_metric,  # Dynamically selected metric (Avg_Value or CPI_Index)
        # title=f"Comparison of {selected_metric.replace('_', ' ').title()} for {selected_class_title} in {selected_year} by Province",
        labels={selected_metric: selected_metric.replace('_', ' ').title()},  # Label based on selected metric
        color='Province',  # Color by Province
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Update axis labels and titles
    fig.update_xaxes(title='Province')
    fig.update_yaxes(title=f'{selected_metric.replace("_", " ").title()} Value')

    # Update layout to improve appearance
    fig.update_layout(
      title=f"Comparison of {selected_metric.replace('_', ' ').title()} for {selected_class_title} in {selected_year} by Province",
      legend_title="Provinces",
      margin=dict(t=50, l=50, b=50, r=50),
      title_x=0.5,  # Center title
      xaxis_title="Province",  # Title for x-axis
      yaxis_title=f"{selected_metric.replace('_', ' ').title()} Index Value",  # Title for y-axis with units
      legend=dict(
          title="Provinces",
          bordercolor="black",  # Set the border color
          borderwidth=2  # Set the border width
      )
  )


    return fig


@app.callback(
    Output('trend-analysis-plot', 'figure'),
    Input('province-dropdown-trend', 'value'),
    Input('metric-dropdown-trend', 'value'),
    Input('method-dropdown', 'value')
)
def update_trend_analysis_chart(province, metric, method):
    # Filter data by province
    province_data = df[df['Province'] == province].sort_values(by='Year')

    # Ensure metric exists and data is not empty
    if metric not in province_data.columns or province_data.empty:
        return px.bar(
            title="No data available for the selected province or metric.",
            labels={'x': 'Year', 'y': 'Metric Value'}
        )

    # Calculate percentage change between years
    province_data['Change'] = province_data[metric].pct_change() * 100  # Convert to percentage

    # Define thresholds for significant changes
    increase_threshold = 20  # 20% increase
    decrease_threshold = -20  # 20% decrease

    # Classify changes as 'Above Trend', 'Below Trend', or 'Normal'
    if method == 'z_score':
        province_data['Trend'] = np.where(province_data['Change'] > increase_threshold,
                                          'Above Trend', 'Normal')
    elif method == 'iqr':
        province_data['Trend'] = np.where(province_data['Change'] < decrease_threshold,
                                          'Below Trend', 'Normal')
    else:
        province_data['Trend'] = 'Normal'

    # Fill missing values with 'Normal'
    province_data['Trend'].fillna('Normal', inplace=True)

    # Create a bar plot with meaningful legends
    fig = px.bar(
        province_data,
        x='Year',
        y=metric,
        color='Trend',
        color_discrete_map={
            'Above Trend': 'green',
            'Below Trend': 'red',
            'Normal': 'blue'
        },
        labels={'Trend': 'Trend Analysis'},
        # title=f"Trend Analysis of {metric} in {province} - Method: {method.capitalize()}"
    )

    # Update layout for legend on the right with border
    fig.update_layout(
        yaxis=dict(title=metric),
        legend=dict(
            title='Legend',
            orientation='v',
            y=0.5,
            x=1.1,
            xanchor='left',
            yanchor='middle',
            bordercolor='black',
            borderwidth=2,
            bgcolor='white'
        ),
        margin=dict(t=40, b=40, l=40, r=80),
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)