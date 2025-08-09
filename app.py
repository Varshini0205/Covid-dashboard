import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Country','Date']).reset_index(drop=True)
    df[['NewConfirmed', 'NewDeaths']] = df.groupby('Country')[['Confirmed', 'Deaths']].diff().fillna(0)
    df['NewConfirmed7'] = df.groupby('Country')['NewConfirmed'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['NewDeaths7'] = df.groupby('Country')['NewDeaths'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['CFR'] = np.where(df['Confirmed'] > 0, df['Deaths'] / df['Confirmed'] * 100, 0)
    # Add Active cases here (Step 2)
    df['Active'] = df['Confirmed'] - df['Recovered'] - df['Deaths']
    return df

df = load_data()

# 2. Sidebar (filters & info)
st.sidebar.header("Filters & Info")

# Date range filter (Step 1)
start_date = st.sidebar.date_input("Start date", df['Date'].min())
end_date = st.sidebar.date_input("End date", df['Date'].max())
if start_date > end_date:
    st.sidebar.error("Error: Start date must be before end date")

df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Country selection (already existing)
countries = df['Country'].unique()
selected_countries = st.sidebar.multiselect("Select countries to compare", countries, default=['India'])

# Metric selection (Step 2: add Active to list)
metric = st.sidebar.selectbox("Select metric", ['Confirmed', 'Deaths', 'NewConfirmed', 'NewConfirmed7', 'CFR', 'Active'])

# Sidebar metric reference table (optional)
metrics_info = {
    "Metric": ["Confirmed", "Deaths", "NewConfirmed", "NewConfirmed7", "CFR", "Active"],
    "Meaning": [
        "Total confirmed COVID-19 cases accumulated so far",
        "Total deaths reported due to COVID-19",
        "Number of new confirmed cases reported each day",
        "7-day moving average of daily new confirmed cases",
        "Case Fatality Rate = (Deaths / Confirmed) * 100",
        "Active cases = Confirmed - Recovered - Deaths"
    ]
}
metrics_df = pd.DataFrame(metrics_info)
st.sidebar.header("Metric Reference")
st.sidebar.table(metrics_df)

# Optional: country flags in sidebar (Step 8)
flags = {
    'India': '🇮🇳',
    'United States': '🇺🇸',
    'Brazil': '🇧🇷',
    # add more flags here if you want
}
st.sidebar.header("Selected Countries with Flags")
for c in selected_countries:
    st.sidebar.markdown(f"{flags.get(c, '')} {c}")

# 3. Filter data by selected countries
filtered_df = df[df['Country'].isin(selected_countries)]

# 4. Summary stats (Step 3)
st.header("Summary Statistics")
for country in selected_countries:
    latest = filtered_df[filtered_df['Country'] == country].sort_values('Date').iloc[-1]
    st.markdown(f"**{country}:** Confirmed: {int(latest['Confirmed'])}, Deaths: {int(latest['Deaths'])}, Active: {int(latest['Active'])}, CFR: {latest['CFR']:.2f}%")

# 5. Plot (with dynamic colors) - you can add dynamic color mapping here
color_list = ['blue', 'green', 'orange', 'red']
color_map = {country: color_list[i % len(color_list)] for i, country in enumerate(selected_countries)}

fig = px.line(
    filtered_df,
    x='Date',
    y=metric,
    color='Country',
    color_discrete_map=color_map,
    title=f"{metric} over time"
)

# Bar chart: Total Confirmed cases per selected country on the latest filtered date
st.subheader("Total Confirmed Cases by Country")

# Get the latest date in the filtered data range
latest_filtered_date = filtered_df['Date'].max()

# Filter data for the latest date in filtered range
latest_filtered_data = filtered_df[filtered_df['Date'] == latest_filtered_date]

# Create bar chart using Plotly Express
fig_bar = px.bar(
    latest_filtered_data,
    x='Country',
    y='Confirmed',
    color='Country',
    title=f"Total Confirmed Cases by Country on {latest_filtered_date.date()}",
    labels={'Confirmed': 'Confirmed Cases'}
)

st.plotly_chart(fig_bar, use_container_width=True)


# 6. Layout with columns (Step 4)
col1, col2 = st.columns([3,1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.dataframe(filtered_df[['Date', 'Country', metric]].head(10))

# 7. Download button (Step 5)
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered data as CSV", csv, "filtered_covid_data.csv", "text/csv")


# 8. Heatmap (Step 6)
latest_date = df['Date'].max()
latest_data = df[df['Date'] == latest_date]
corr = latest_data[['Confirmed', 'Deaths', 'NewConfirmed', 'NewDeaths', 'Active', 'CFR']].corr()

fig2, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
plt.title(f'Correlation Heatmap ({latest_date.date()})')
st.pyplot(fig2)
