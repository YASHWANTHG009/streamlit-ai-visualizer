import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Data Visualizer", layout="wide")

st.title("📊 AI-Driven Data Cleaning & Visualization")

# 1️⃣ Upload CSV

uploaded_file = st.file_uploader("data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv("uploaded_file")
    st.success("File loaded successfully!")

    # 2️⃣ Raw Data Preview

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(50))

    # 3️⃣ Data Cleaning

    st.subheader("🧹 Data Cleaning")

    null_count = df.isnull().sum()
    st.write("Null values per column:")
    st.dataframe(null_count[null_count > 0])

    if st.checkbox("Fill NULL values"):
        df = df.fillna({
            col: df[col].median() if df[col].dtype != "object" else "UNKNOWN"
            for col in df.columns
        })
        st.success("NULL values filled intelligently!")

    # 4️⃣ Column Type Detection

    st.subheader("🧠 Column Type Detection")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    col1, col2 = st.columns(2)
    col1.write("Numeric Columns")
    col1.write(numeric_cols)

    col2.write("Text Columns")
    col2.write(text_cols)

    # 5️⃣ Price Segregation

    st.subheader("💰 Price Segregation")

    price_cols = [c for c in df.columns if "price" in c.lower()]

    st.write("Detected price-related columns:")
    st.write(price_cols)

    # 6️⃣ Visualizations
    
    st.subheader("📈 Visualizations")

    # ---- 6.1 Distribution
    st.markdown("### Price Distribution")
    selected_price = st.selectbox("Select price column", price_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_price], kde=True, ax=ax)
    st.pyplot(fig)

    # ---- 6.2 Boxplot
    st.markdown("### Price Spread (Outliers)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(y=df[selected_price], ax=ax2)
    st.pyplot(fig2)

    # ---- 6.3 Product-wise Avg Price
    if "product" in df.columns:
        st.markdown("### Product-wise Average Price")
        avg_price = df.groupby("product")[selected_price].mean().sort_values(ascending=False)

        fig3, ax3 = plt.subplots(figsize=(10,5))
        avg_price.head(20).plot(kind="bar", ax=ax3)
        st.pyplot(fig3)
    st.markdown("### Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # 7️⃣ Smart Filters
    
    st.subheader("🎯 Smart Filtering")

    if selected_price:
        min_val, max_val = st.slider(
            "Filter by price range",
            float(df[selected_price].min()),
            float(df[selected_price].max()),
            (float(df[selected_price].min()), float(df[selected_price].max()))
        )

        filtered_df = df[
            (df[selected_price] >= min_val) &
            (df[selected_price] <= max_val)
        ]

        st.write("Filtered Data")
        st.dataframe(filtered_df.head(50))

   
    # 8️⃣ Download Cleaned Data
    
    st.subheader("⬇️ Download Cleaned Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )
