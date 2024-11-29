import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from prophet import Prophet
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="🌌 Advanced Global Warming Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #f5f5f5;
    }
    .stApp {
        background-color: #121212;
    }
    h1, h2, h3, h4 {
        color: #f5f5f5;
    }
    .stSidebar {
        background-color: #1f1f1f;
    }
    .stSidebar .st-radio {
        color: #f5f5f5;
    }
    .css-1aumxhk {
        background-color: #1f1f1f;
    }
    .stDataFrame {
        border: 1px solid #f5f5f5;
    }
    .css-1ekf893 {
        color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.title("✨ Navigation Menu")
menu_options = [
    "🏠 Home",
    "📊 Scenario Analysis",
    "📈 Advanced Visualizations",
    "🔮 Time Series Forecast (ARIMA & Prophet)",
    "📥 Upload & Analyze Data",
    "📋 Generate Reports",
    "ℹ️ About"
]
menu_choice = st.sidebar.radio("Navigate", menu_options)

# Load the dataset
@st.cache_data
def load_data(file_path="fully_cleaned_global_warming_sim_dataset.csv"):
    return pd.read_csv(file_path)

df = load_data()

# Function to generate custom scenario predictions
def generate_scenario(df, co2_change, ch4_change, n2o_change):
    scenario_df = df.copy()
    scenario_df["CO2_Concentration_ppm"] += co2_change
    scenario_df["CH4_Concentration_ppb"] += ch4_change
    scenario_df["N2O_Concentration_ppb"] += n2o_change
    X = scenario_df[["CO2_Concentration_ppm", "CH4_Concentration_ppb", "N2O_Concentration_ppb"]]
    y = scenario_df["Temperature_Anomaly_C"]
    model = LinearRegression()
    model.fit(X, y)
    scenario_df["Predicted_Temperature_Anomaly_C"] = model.predict(X)
    return scenario_df

# Home Page
if menu_choice == "🏠 Home":
    st.title("🌌 Advanced Global Warming Analysis")
    st.markdown("""
    <h3>Welcome to the Advanced Global Warming Analysis Tool</h3>
    <p>This platform allows you to explore and analyze climate data interactively with advanced tools and visualizations.</p>
    """, unsafe_allow_html=True)
    st.image("https://cdn.mos.cms.futurecdn.net/6ZW3VY5dZJbYSD7FeAsKe6-1200-80.jpg", use_column_width=True)
    st.write("### Dataset Preview")
    st.dataframe(df.head(10))

# Scenario Analysis
elif menu_choice == "📊 Scenario Analysis":
    st.header("📊 Advanced Scenario Analysis")

    st.write("### Customize Greenhouse Gas Changes:")
    co2_change = st.slider("CO2 Change (ppm)", -10.0, 10.0, 0.0, step=0.5)
    ch4_change = st.slider("CH4 Change (ppb)", -50.0, 50.0, 0.0, step=5.0)
    n2o_change = st.slider("N2O Change (ppb)", -5.0, 5.0, 0.0, step=0.5)

    scenario_df = generate_scenario(df, co2_change, ch4_change, n2o_change)

    st.write("### Scenario Results")
    fig = px.line(
        scenario_df,
        x="Year",
        y=["Temperature_Anomaly_C", "Predicted_Temperature_Anomaly_C"],
        labels={"value": "Temperature Anomaly (°C)", "variable": "Scenario"},
        title="Scenario Analysis of Temperature Anomalies"
    )
    st.plotly_chart(fig)

# Advanced Visualizations
elif menu_choice == "📈 Advanced Visualizations":
    st.header("📈 Advanced Visualizations")

    # Altair Scatter Plot
    st.write("### Altair Interactive Scatter Plot")
    alt_chart = alt.Chart(df).mark_circle(size=60).encode(
        x='CO2_Concentration_ppm',
        y='Temperature_Anomaly_C',
        color='Year:N',
        tooltip=['Year', 'CO2_Concentration_ppm', 'Temperature_Anomaly_C']
    ).interactive()
    st.altair_chart(alt_chart, use_container_width=True)

    # Heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

# Time Series Forecast
elif menu_choice == "🔮 Time Series Forecast (ARIMA & Prophet)":
    st.markdown("""
    <style>
    .forecast-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .forecast-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .forecast-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .forecast-chart {
        margin-top: 20px;
        background: #1f1f1f;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        font-size: 1rem;
    }
    .analyze-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .analyze-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    </style>
    """, unsafe_allow_html=True)

    # Forecasting Başlığı
    st.markdown("""
    <div class="forecast-container">
        <h2>🔮 Time Series Forecasting</h2>
        <p>
            Analyze and predict future temperature anomalies using ARIMA and Prophet models.<br>
            These forecasts are designed to help visualize long-term climate trends interactively.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ARIMA Forecast
    st.write("### ARIMA Forecast")
    arima_model = ARIMA(df["Temperature_Anomaly_C"], order=(2, 1, 2))
    arima_result = arima_model.fit()
    forecast_years = 50
    forecast_index = pd.date_range(start="2025", periods=forecast_years, freq="YE")
    forecast = arima_result.forecast(steps=forecast_years)

    # ARIMA Grafiği
    st.markdown("""
    <div class="forecast-chart">
    """, unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Year"], y=df["Temperature_Anomaly_C"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast_index.year, y=forecast, mode="lines", name="Forecast"))
    fig.update_layout(
        title="ARIMA Forecast (Next 50 Years)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prophet Forecast
    st.write("### Prophet Forecast")
    prophet_df = df.rename(columns={"Year": "ds", "Temperature_Anomaly_C": "y"})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=50, freq="YE")
    forecast = prophet_model.predict(future)

    # Prophet Grafiği
    st.markdown("""
    <div class="forecast-chart">
    """, unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Actual"))
    fig2.add_trace(go.Scatter(x=future["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig2.update_layout(
        title="Prophet Forecast (Next 50 Years)",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (°C)",
        template="plotly_dark"
    )
    st.plotly_chart(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

    # Özet ve Gelecek Planları
    st.markdown("""
    <div class="forecast-container">
        <p>
            These time series models provide valuable insights into long-term climate trends.<br>
            Further improvements, including model tuning and additional forecasting metrics, are planned for future releases.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Upload and Analyze Data
elif menu_choice == "📥 Upload & Analyze Data":
    st.markdown("""
    <style>
    .upload-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .upload-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .upload-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .data-preview {
        margin-top: 20px;
        background: #1f1f1f;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        font-size: 1rem;
        text-align: left;
        overflow-x: auto;
    }
    .analyze-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .analyze-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    .no-data-warning {
        color: #ff6666;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Veri Yükleme Başlığı
    st.markdown("""
    <div class="upload-container">
        <h2>📥 Upload and Analyze Your Data</h2>
        <p>
            Upload your CSV dataset to perform interactive analysis.  
            The uploaded dataset will be previewed, and summary statistics will be generated for further insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dosya Yükleme Aracı
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file:
        # Yüklenen Dosya ile Çalışma
        user_df = pd.read_csv(uploaded_file)

        # Veri Seti Önizlemesi
        st.write("### Uploaded Dataset Preview")
        st.markdown("""
        <div class="data-preview">
        """, unsafe_allow_html=True)
        st.dataframe(user_df.head())
        st.markdown("</div>", unsafe_allow_html=True)

        # Veri Seti Tanımı
        st.write("### Dataset Description")
        st.markdown("""
        <div class="data-preview">
        """, unsafe_allow_html=True)
        st.write(user_df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

        # Ek Analiz Seçenekleri
        st.write("### Explore Data Further")
        analyze_choice = st.selectbox(
            "Choose an analysis option:",
            ["Correlation Heatmap", "Histogram", "Scatter Plot"]
        )

        # Korelasyon Isı Haritası
        if analyze_choice == "Correlation Heatmap":
            st.write("#### Correlation Heatmap")
            corr = user_df.corr()
            st.write("Correlation Matrix:", corr)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # Histogram
        elif analyze_choice == "Histogram":
            st.write("#### Histogram")
            column_to_plot = st.selectbox("Choose a column for the histogram:", user_df.columns)
            bins = st.slider("Number of bins:", 5, 50, 20)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            user_df[column_to_plot].hist(bins=bins, ax=ax, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogram of {column_to_plot}")
            ax.set_xlabel(column_to_plot)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        # Scatter Plot
        elif analyze_choice == "Scatter Plot":
            st.write("#### Scatter Plot")
            x_col = st.selectbox("Select X-axis column:", user_df.columns)
            y_col = st.selectbox("Select Y-axis column:", user_df.columns)
            st.markdown("""
            <div class="data-preview">
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            user_df.plot.scatter(x=x_col, y=y_col, ax=ax, color="orange")
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="no-data-warning">
            Please upload a CSV file to begin analysis.
        </div>
        """, unsafe_allow_html=True)

# Generate Reports
elif menu_choice == "📋 Generate Reports":
    st.markdown("""
    <style>
    .report-container {
        background: linear-gradient(to bottom, #1c1c1c, #121212);
        color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.8);
    }
    .report-container h2 {
        font-size: 2.5rem;
        color: #00ff8a;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .report-container p {
        font-size: 1.2rem;
        color: #dcdcdc;
        line-height: 1.8;
    }
    .report-button {
        background: #00d4ff;
        color: #000000;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, background 0.3s ease;
        cursor: pointer;
    }
    .report-button:hover {
        transform: scale(1.1);
        background: #00ff8a;
    }
    </style>
    """, unsafe_allow_html=True)

    # Raporlama Bölümü
    st.markdown("""
    <div class="report-container">
        <h2>📋 Generate Reports</h2>
        <p>
            Export the dataset in various formats, including CSV, Excel, and PDF.<br>
            Download and analyze the reports locally for deeper insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Veri Setini CSV Formatında İndirme
    buffer_csv = BytesIO()
    df.to_csv(buffer_csv, index=False)
    buffer_csv.seek(0)

    # Excel İndir
    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="GlobalWarmingData")
    buffer_excel.seek(0)

    # İndirilebilir Seçenekler
    st.write("### Download Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="Download CSV",
            data=buffer_csv,
            file_name="global_warming_analysis.csv",
            mime="text/csv",
            help="Download the dataset in CSV format.",
            key="csv_download"
        )
    with col2:
        st.download_button(
            label="Download Excel",
            data=buffer_excel,
            file_name="global_warming_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the dataset in Excel format.",
            key="excel_download"
        )
    with col3:
        st.button(
            label="Generate PDF (Coming Soon)",
            help="This feature will allow you to generate a PDF report. Stay tuned!",
            key="pdf_coming_soon",
            disabled=True
        )

    # Ek Raporlama ve Bilgilendirme
    st.markdown("""
    <div class="report-container">
        <p>
            Note: PDF report generation is under development. Soon, you will be able to create detailed analytical reports 
            that include visualizations and insights.
        </p>
    </div>
    """, unsafe_allow_html=True)


# About Page
elif menu_choice == "ℹ️ About":
    st.markdown("""
    <style>
    /* Genel Stil */
    body {
        background: linear-gradient(135deg, #1a1a1a, #121212);
        color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }
    .about-container {
        background: linear-gradient(135deg, #212121, #1a1a1a);
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.7);
        color: #ffffff;
        max-width: 900px;
        margin: auto;
        text-align: center;
    }
    .about-container h2 {
        color: #00c8ff;
        font-size: 28px;
        margin-bottom: 15px;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.8);
    }
    .about-container p {
        color: #cccccc;
        line-height: 1.8;
        margin-bottom: 20px;
        font-size: 16px;
    }
    .tech-box {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 15px;
    }
    .tech-item {
        background: #1f1f1f;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #444;
        text-align: center;
        font-size: 14px;
        color: #ffffff;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .tech-item:hover {
        background: #00c8ff;
        color: #1a1a1a;
        transform: translateY(-5px);
    }
    .developer-box {
        margin-top: 30px;
        padding: 20px;
        background: #1f1f1f;
        border-radius: 10px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.6);
    }
    .developer-box h3 {
        color: #00ff8a;
        margin-bottom: 15px;
    }
    .developer-box p {
        color: #e0e0e0;
        margin: 5px 0;
    }
    .contact-link {
        color: #00c8ff;
        text-decoration: none;
        font-weight: bold;
    }
    .contact-link:hover {
        text-decoration: underline;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #cccccc;
        font-size: 14px;
        padding-top: 20px;
        border-top: 1px solid #444;
    }
    .footer a {
        color: #00c8ff;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sayfa Başlığı ve Giriş
    st.markdown("""
    <div class='about-container'>
        <h2>🌍 Advanced Global Warming Analysis</h2>
        <p>
        This application provides an advanced platform to analyze and forecast global warming trends interactively.  
        It offers tools for data visualization, scenario simulation, and predictive analytics, designed to deliver meaningful insights.
        </p>
        <div class='developer-box'>
            <h3>👩‍💻 Developer: Pınar Topuz</h3>
            <p>📍 Samsun, Turkey</p>
            <p>🎓 Electronics & Communication Technology</p>
            <p>🌟 Interests: Climate Data Analysis, AI in Environmental Sciences</p>
            <p>
                📫 <a href='mailto:piinartp@gmail.com' class='contact-link'>Email</a> | 
                <a href='https://github.com/your-repo' class='contact-link'>GitHub</a> | 
                <a href='https://www.linkedin.com/piinartp' class='contact-link'>LinkedIn</a>
            </p>
        </div>
        <h3>🎯 Project Goals</h3>
        <ul style='text-align: left; margin-left: 40px;'>
            <li>Provide an intuitive platform for analyzing global warming trends.</li>
            <li>Enable users to simulate different environmental scenarios.</li>
            <li>Offer tools for advanced visualizations and dynamic forecasting.</li>
        </ul>
        <h3>🛠️ Technologies Used</h3>
        <div class='tech-box'>
            <div class='tech-item'>Python</div>
            <div class='tech-item'>Streamlit</div>
            <div class='tech-item'>Plotly</div>
            <div class='tech-item'>Altair</div>
            <div class='tech-item'>Matplotlib</div>
            <div class='tech-item'>Seaborn</div>
            <div class='tech-item'>Prophet</div>
            <div class='tech-item'>ARIMA</div>
        </div>
        <h3>🚀 Future Developments</h3>
        <ul style='text-align: left; margin-left: 40px;'>
            <li>Integration with live environmental data APIs (e.g., NASA, NOAA).</li>
            <li>Advanced forecasting models using deep learning techniques.</li>
            <li>Interactive global map with real-time data visualization.</li>
            <li>Automatic PDF report generation with detailed analysis.</li>
        </ul>
    </div>
    <div class='footer'>
        <p>Thank you for exploring this application!</p>
        <p>🔗 <a href='https://github.com/your-repo'>GitHub Repository</a> | 
        <a href='mailto:piinartp@gmail.com'>Contact Developer</a></p>
    </div>
    """, unsafe_allow_html=True)
