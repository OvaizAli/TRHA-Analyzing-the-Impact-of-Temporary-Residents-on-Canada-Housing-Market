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
    Avg_Housing_Value = row['Avg_Housing_Value']
    workers_count = row['Total_Workers_Count']
    cpi = row['Avg_Total_CPI']

    if Avg_Housing_Value > 0.5 and workers_count < 0:
        cluster_descriptions[row['Cluster']] = f"Cluster {row['Cluster']}: High Housing Prices, Low Workers"
    elif Avg_Housing_Value < 0 and workers_count > 0:
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
        html.H2("Forecasting Average Housing Price Index Value by Province"),
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
                    value='Ontario',  # default value
                    clearable=False,
                    style={'width': '100%'}
                ),
            ], style={'display': 'inline-block', 'width': '30%', 'padding': '10px'}),

            html.Div([
                html.Label("Select Feature for X-axis"),
                dcc.Dropdown(
                    id='x-feature-dropdown',
                    options=[
                        {'label': 'Number_of_Study_Permits_Issued', 'value': 'Number_of_Study_Permits_Issued'},
                        {'label': 'Study_Permit_To_Value_Impact_Metric', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total_Workers_Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Workers_To_Housing_Impact_Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Avg_Total_CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Avg_CPI_TRIM', 'value': 'Avg_CPI_TRIM'}
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
                        {'label': 'Number_of_Study_Permits_Issued', 'value': 'Number_of_Study_Permits_Issued'},
                        {'label': 'Study_Permit_To_Value_Impact_Metric', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total_Workers_Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Workers_To_Housing_Impact_Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Avg_Total_CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Avg_CPI_TRIM', 'value': 'Avg_CPI_TRIM'}
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
                        {'label': 'Avg_Housing_Value', 'value': 'Avg_Housing_Value'},
                        {'label': 'Study_Permit_To_Value_Impact_Metric', 'value': 'Study_Permit_To_Value_Impact_Metric'},
                        {'label': 'Total_Workers_Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Number_of_Class_Titles', 'value': 'Number_of_Class_Titles'},
                        {'label': 'Workers_To_Housing_Impact_Ratio', 'value': 'Workers_To_Housing_Impact_Ratio'},
                        {'label': 'Combined_Impact_Ratio', 'value': 'Combined_Impact_Ratio'},
                        {'label': 'Avg_Total_CPI', 'value': 'Avg_Total_CPI'},
                        {'label': 'Avg_Total_CPI_Seasonally_Adjusted', 'value': 'Avg_Total_CPI_Seasonally_Adjusted'},
                        {'label': 'Avg_CPI_MEDIAN', 'value': 'Avg_CPI_MEDIAN'}
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
        html.H1("Impact of Occupation on Housing Demand in Canada", style={'textAlign': 'left', 'padding': '20px'}),

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
                        {'label': 'Housing Price Index (Avg_Housing_Value)', 'value': 'Avg_Housing_Value'},
                        {'label': 'Total_Workers_Count', 'value': 'Total_Workers_Count'},
                        {'label': 'Combined_Impact_Ratio', 'value': 'Combined_Impact_Ratio'}
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
    ], style={'padding': '20px'}),  # End of new plot section
])

# Callback for Time Series Forecasting
@app.callback(
    Output("avg-value-time-series", "figure"),
    [
        Input("province-dropdown-1", "value"),
        Input("forecast-years-slider", "value")
    ]
)
def update_time_series_forecast(selected_province, forecast_years):
    df_province_ts = df_grouped_ts[df_grouped_ts['Province'] == selected_province]
    model = ARIMA(df_province_ts['Avg_Housing_Value'], order=(5, 1, 0), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_years)

    # Create the plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_province_ts['Year'], y=df_province_ts['Avg_Housing_Value'], mode='lines+markers', name='Historical Data'))
    fig.add_trace(go.Scatter(x=np.arange(df_province_ts['Year'].max(), df_province_ts['Year'].max() + forecast_years),
                             y=forecast, mode='lines+markers', name='Forecast', line=dict(dash='dash')))

    fig.update_layout(title=f"Average Housing Index Value Forecast for {selected_province}",
                      xaxis_title="Year", yaxis_title="Average Housing Value")
    
    return fig


# Callback for Feature Relationship Scatter Plot
@app.callback(
    Output("feature-relationship-plot", "figure"),
    [
        Input("province-dropdown-2", "value"),
        Input("x-feature-dropdown", "value"),
        Input("y-feature-dropdown", "value"),
        Input("size-feature-dropdown", "value"),
        Input("year-range-slider", "value")
    ]
)
def update_scatter_plot(selected_province, x_feature, y_feature, size_feature, year_range):
    df_province = df[(df['Province'] == selected_province) &
                     (df['Year'] >= year_range[0]) &
                     (df['Year'] <= year_range[1])]
    
    fig = px.scatter(df_province, x=x_feature, y=y_feature, size=size_feature,
                     color="Province", hover_data=["Year", "Avg_Housing_Value"],
                     title=f"{x_feature} vs {y_feature} for {selected_province}")

    fig.update_layout(title=f"Selected Feature(s) Relationship for {selected_province}")
    return fig


# Callback for updating the scatter plot and radar chart for clustering
@app.callback(
    Output("regional-cluster-plot", "figure"),
    [
        Input("province-dropdown", "value"),
        Input("chart-toggle", "value")
    ]
)
def update_cluster_plot(selected_province, chart_type):
    df_province = df_grouped[df_grouped['Province'] == selected_province]

    if chart_type == 'scatter':
        df_province['Cluster'] = df_province['Cluster'].astype(str)
        df_province['hover_text'] = df_province.apply(lambda row: f"Cluster: {cluster_descriptions[int(row['Cluster'])]}<br>Avg Value: {row['Avg_Housing_Value']:.2f}<br>Workers Count: {row['Total_Workers_Count']:.2f}<br>CPI: {row['Avg_Total_CPI']:.2f}<br>Year: {row['Year']}", axis=1)
        
        fig = px.scatter(df_province,
                         x='PCA1', y='PCA2', color='Cluster',
                         title=f"Clustered Regional Housing Trends for {selected_province}",
                         labels={'PCA1': 'Economic Dimension 1', 'PCA2': 'Economic Dimension 2'},
                         hover_name='Province',
                         hover_data={'Year': True, 'Avg_Housing_Value': True, 'Total_Workers_Count': True, 'Avg_Total_CPI': True},
                         color_discrete_sequence=px.colors.qualitative.Set1)
        
        fig.for_each_trace(lambda t: t.update(name=cluster_descriptions[int(t.name)]))
        cluster_sizes = df_province['Cluster'].value_counts().to_dict()
        df_province['point_size'] = df_province['Cluster'].map(cluster_sizes).apply(lambda x: x * 12)
        fig.update_traces(marker=dict(size=df_province['point_size'], opacity=0.7, line=dict(width=0)))

        annotations = []
        for i, row in df_province.iterrows():
            annotations.append(
                dict(x=row['PCA1'], y=row['PCA2'], text=f"{row['Year']}", showarrow=False, font=dict(size=12, color='black')))
        
        fig.update_layout(annotations=annotations)
        fig.update_layout(template="plotly_white", xaxis_title="Economic Dimension 1 (PCA1)", yaxis_title="Economic Dimension 2 (PCA2)")

    elif chart_type == 'radar':
        cluster_means = df_province.groupby('Cluster')[features].mean().reset_index()
        fig = go.Figure()

        for cluster in cluster_means['Cluster']:
            fig.add_trace(go.Scatterpolar(
                r=cluster_means[cluster_means['Cluster'] == cluster][features].values.flatten(),
                theta=features,
                fill='toself',
                name=cluster_descriptions[cluster]
            ))

        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
                          title=f"Radar Chart of Cluster Characteristics for {selected_province}")
    
    return fig


# New callback for "Occupational Influence on Housing Demand" plot
@app.callback(
    Output("occupational-influence-plot", "figure"),
    [Input("class-title-dropdown", "value"),
     Input("metric-dropdown", "value"),
     Input("year-dropdown", "value")]
)
def update_occupational_influence_plot(selected_class_title, selected_metric, selected_year):
    # Filter data by class title and year
    df_filtered_class_year = df[(df['Most_Common_Class_Title'] == selected_class_title) &
                                (df['Year'] == selected_year)]
    
    # Aggregate by Province with mean for the selected metric
    df_avg_metric = df_filtered_class_year.groupby('Province', as_index=False)[selected_metric].mean()
    
    # Plotting the bar chart
    fig = px.bar(
        df_avg_metric,
        x='Province',
        y=selected_metric,
        title=f"{selected_metric.replace('_', ' ').title()} for {selected_class_title} in {selected_year} by Province",
        labels={selected_metric: selected_metric.replace('_', ' ').title()},
        color='Province',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    # Updating layout and axis titles
    fig.update_xaxes(title='Province')
    fig.update_yaxes(title=f'{selected_metric.replace("_", " ").title()} Value')

    fig.update_layout(
        title=f"{selected_metric.replace('_', ' ').title()} for {selected_class_title} in {selected_year} by Province",
        legend_title="Provinces",
        margin=dict(t=50, l=50, b=50, r=50),
        xaxis_title="Province",
        yaxis_title=f"{selected_metric.replace('_', ' ').title()} Index Value",
    )

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
