import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
with open('rf_model3.pkl', 'rb') as file:
    model = pickle.load(file)

# Define spending categories
spending_categories = {1: "Low Spender", 2: "Medium Spender", 0: "High Spender"}

# Function to predict spending category
def predict_spending_category(age, gender, total_income, total_spendings):
    gender_encoded = 1 if gender == "Male" else 0  # Encoding gender
    features = [[age, gender_encoded, total_income,total_spendings]]
    prediction = model.predict(features)[0]
    return spending_categories[prediction]

# Function for recommendations
def recommend_products(spending_category):
    recommendations = {
        "Low Spender": ["Discount Coupons", "Budget Phones", "Affordable Fashion"],
        "Medium Spender": ["Premium Gadgets", "Branded Clothing", "Dining Offers"],
        "High Spender": ["Luxury Watches", "Exclusive Memberships", "High-End Electronics"]
    }
    return recommendations.get(spending_category, [])

# Streamlit UI
st.set_page_config(page_title="Customer Spending Segmentation", page_icon="🛍️", layout="wide")
st.title("🛍️ Customer Spending Segmentation")
st.write("Enter customer details to predict their spending category.")

# Sidebar for user inputs
st.sidebar.header("User Input")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
total_income = st.sidebar.number_input("Total Income ($)", min_value=500, max_value=200000, value=50000)
total_spendings = st.sidebar.number_input("Total Spendings ($)", min_value=500, max_value=200000, value=50000)
# Predict button
if st.sidebar.button("Predict Spending Category"):
    category = predict_spending_category(age, gender, total_income, total_spendings)
    st.sidebar.success(f"Predicted Spending Category: **{category}**")
    
    # Show recommendations
    st.sidebar.subheader("🎯 Recommended Products for You")
    for item in recommend_products(category):
        st.sidebar.write(f"- {item}")

st.subheader("📂 Upload Dataset for Analysis")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Ensure required columns exist
    required_columns = {"Age", "Gender", "Total Income", "Total Spendings"}
    if not required_columns.issubset(df.columns):
        st.error("❌ The dataset must contain 'Age', 'Gender', 'Total Spendings and 'Total Income' columns.")
    else:
        # Convert Gender to numerical (if needed)
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

        # Predictions for entire dataset
        df["Spending Category"] = model.predict(df[["Age", "Gender", "Total Income","Total Spendings"]])
        df["Spending Category"] = df["Spending Category"].map(spending_categories)

        st.write("### 📊 Spending Category Predictions")
        st.dataframe(df[["Age", "Gender", "Total Income", "Spending Category", "Total Spendings"]].head())

        # ---- VISUALIZATION ----
        st.subheader("📊 Spending Insights")

        # Histograms for Age and Total Income
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Age"], bins=10, kde=True, color="blue", ax=axes[0])
        axes[0].set_title("Age Distribution")

        sns.histplot(df["Total Income"], bins=10, kde=True, color="green", ax=axes[1])
        axes[1].set_title("Total Income Distribution")

        st.pyplot(fig)

        # Spending Category Distribution
        st.subheader("📌 Spending Category Breakdown")
        category_counts = df["Spending Category"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=category_counts.index, y=category_counts.values, palette="coolwarm", ax=ax)
        ax.set_title("Spending Category Distribution")
        st.pyplot(fig)

st.write("This app provides insights into customer spending behavior based on input data and suggests tailored products to enhance their experience.")
# Visualization Section
st.subheader("📊 Spending Trends & Insights")
st.write("Understanding spending patterns through data visualization.")

# Dummy data for visualization
data = {
    "Age": [20, 25, 30, 35, 40, 45, 50, 55, 60],
    "Total Income": [15000, 25000, 35000, 45000, 60000, 75000, 90000, 120000, 150000],
    "Spending Score": [20, 40, 60, 80, 50, 30, 70, 90, 100],
    "Total Spending":[3000,12000, 4500, 68000, 3000, 4000, 5000, 3200, 40000]
}
df = pd.DataFrame(data)

# Histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df["Age"], bins=10, kde=True, color="blue", ax=axes[0])
axes[0].set_title("Age Distribution")
sns.histplot(df["Total Income"], bins=10, kde=True, color="green", ax=axes[1])
axes[1].set_title("Total Income Distribution")
st.pyplot(fig)

# Bar chart for spending categories
st.subheader("📌 Spending Category Breakdown")
category_counts = pd.Series(spending_categories.values()).value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="coolwarm", ax=ax)
ax.set_title("Spending Category Distribution")
st.pyplot(fig)

st.write("This app provides insights into customer spending behavior based on input data and suggests tailored products to enhance their experience.")