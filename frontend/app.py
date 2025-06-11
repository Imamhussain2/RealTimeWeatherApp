# ============ TAB 1: ANALYTICS ============
with tab1:
    st.subheader("📡 Current Weather Stats Across Major Cities")

    if st.button("📥 Load Weather Data"):
        df = fetch_data()
        if df is not None:
            st.success("✅ Data successfully loaded!")

            # Raw Data
            with st.expander("🔍 Raw Weather Data"):
                st.dataframe(df)

            # Summary Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Avg Temp (°C)", round(df["Temperature (°C)"].mean(), 1))
            col2.metric("💧 Avg Humidity (%)", round(df["Humidity (%)"].mean(), 1))
            col3.metric("🍃 Avg Wind Speed (m/s)", round(df["Wind Speed (m/s)"].mean(), 1))

            # Convert time column for plots
            df["Data Time"] = pd.to_datetime(df["Data Time"])

            # Temperature Line Chart
            st.markdown("### 🕒 Temperature Trends")
            fig_line = px.line(df.sort_values("Data Time"), x="Data Time", y="Temperature (°C)", color="City")
            st.plotly_chart(fig_line, use_container_width=True)

            # Temperature Analysis
            st.markdown("### 🔥 Temperature Analysis")
            fig_temp = px.bar(df, x="City", y="Temperature (°C)", color="Temperature (°C)",
                              text="Temperature (°C)", color_continuous_scale='thermal')
            st.plotly_chart(fig_temp, use_container_width=True)

            # Humidity Box Plot
            st.markdown("### 📦 Humidity Distribution by City")
            fig_box = px.box(df, x="City", y="Humidity (%)", color="City")
            st.plotly_chart(fig_box, use_container_width=True)

            # Weather Pie Chart
            st.markdown("### 🌈 Weather Distribution")
            fig_pie = px.pie(df, names="Weather Description", title="Weather Types",
                             color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Scatter Plot
            st.markdown("### 🔁 Temperature vs Humidity Scatter")
            fig_scatter = px.scatter(df, x="Temperature (°C)", y="Humidity (%)", color="City",
                                     size="Wind Speed (m/s)", hover_name="City",
                                     color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Correlation Heatmap
            st.markdown("### 🧠 Feature Correlation Heatmap")
            numeric_cols = df.select_dtypes(include='number')
            corr = numeric_cols.corr()
            fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Wind Rose (Polar plot)
            st.markdown("### 🧭 Wind Direction Intensity")
            fig_polar = px.bar_polar(df, r="Wind Speed (m/s)", theta="Wind Dir (°)",
                                     color="City", template="plotly_dark", opacity=0.8)
            st.plotly_chart(fig_polar, use_container_width=True)

            # Country-level weather
            st.markdown("### 🌍 Country-wise Weather Summary")
            fig_country = px.sunburst(df, path=["Country", "City", "Weather"], values="Temperature (°C)",
                                      color="Humidity (%)", color_continuous_scale="Viridis")
            st.plotly_chart(fig_country, use_container_width=True)
