# TRHA-Analyzing-the-Impact-of-Temporary-Residents-on-Canada-Housing-Market

This project analyzes the impact of temporary residents, such as international students and foreign workers, on Canada's housing market. By integrating machine learning, advanced visualizations, and a fine-tuned Large Language Model (LLM), we provide actionable insights into housing affordability and availability across Canadian regions.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset Details](#dataset-details)
6. [Machine Learning Techniques](#machine-learning-techniques)
7. [Visualization Modules](#visualization-modules)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Future Improvements](#future-improvements)
10. [Team Contributions](#team-contributions)

---

## **Overview**

The project explores key questions:
- How do factors like study permits, workers, and economic metrics influence housing prices?
- Can historical data predict future housing trends across provinces?
- Which regions exhibit similar economic and housing characteristics?

We analyze relationships between immigration, economic factors, and housing trends through:
- **Advanced Visualizations**: Scatter plots, bar charts, radar charts, and clustering plots.
- **Machine Learning**: Dimensionality reduction (PCA), clustering (k-Means), and time-series forecasting (ARIMA).
- **LLM Insights**: Automated analysis of economic metrics and housing trends for stakeholders.

---

## **Features**

- **Interactive Dashboard**: Analyze housing trends and economic factors regionally.
- **Predictive Modeling**: Time-series forecasting of housing prices using ARIMA.
- **Clustering Analysis**: Regional grouping based on economic characteristics.
- **LLM Integration**: Generates summaries, insights, and visual explanations for stakeholders.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/OvaizAli/TRHA-Analyzing-the-Impact-of-Temporary-Residents-on-Canada-Housing-Market.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

1. Preprocess datasets:
   ```bash
   python preprocess_data.py
   ```
2. Train machine learning models:
   ```bash
   python train_models.py
   ```
3. Run the dashboard:
   ```bash
   python app-llm.py
   ```
4. Access the dashboard locally:
   ```
   http://127.0.0.1:8050/
   ```

---

## **Dataset Details**

- **New Housing Price Index (1981–2024):** Tracks monthly price indices for houses and land.
- **Study Permit Holders Dataset (2015–2024):** Analyzes international student influx regionally.
- **TFWP & IMP Work Permit Dataset (2015–2024):** Provides details on foreign worker occupations and regions.
- **Consumer Price Index (1995–2024):** Reflects inflationary trends and seasonal adjustments.
- **Canadian Interest Rates (2015–2024):** Includes bank rates, mortgage rates, and effective household interest rates.

---

## **Machine Learning Techniques**

1. **Principal Component Analysis (PCA)**
   - **Purpose:** Dimensionality reduction to simplify high-dimensional data.
   - **Key Outputs:**
     - Retained 85% variance using two principal components.
     - Simplified clustering and regional comparisons.

2. **k-Means Clustering**
   - **Purpose:** Group Canadian provinces into clusters based on economic factors.
   - **Key Outputs:**
     - Four clusters revealed distinct housing trends.

3. **Time Series Analysis (ARIMA)**
   - **Purpose:** Forecast future housing prices.
   - **Key Features:**
     - Integrated CPI as a regressor to improve predictions.

---

## **Visualization Modules**

1. **Scatter Plots**
   - **Purpose:** Highlight correlations between variables (e.g., study permits and housing prices).
   - **Example:** Shows how study permits drive housing demand.

2. **Line Graphs**
   - **Purpose:** Depict temporal trends (e.g., housing prices vs. CPI).

3. **Bar Charts**
   - **Purpose:** Compare metrics like housing prices across provinces.

4. **Radar Charts**
   - **Purpose:** Visualize multi-dimensional factors (e.g., regional economic trends).

---

## **Evaluation Metrics**

- **PCA Explained Variance:** 85% of variance retained.
- **Clustering Quality:** Evaluated using Silhouette Score.
- **Forecasting Accuracy:** Assessed with RMSE for ARIMA predictions.

---

## **Future Improvements**

- **Real-Time Data Integration:** Add live feeds for economic metrics.
- **Advanced Forecasting Models:** Experiment with LSTM for housing price predictions.
- **Expanded Analysis:** Include rural regions for comprehensive insights.
- **Enhanced Dashboard:** Refine user experience with more interactive filters.