"""
Extended Supervised & Unsupervised Learning GUI with Organized Layout
======================================================================
This Streamlit app provides a comprehensive suite of operations for:
- Exploratory Data Analysis (EDA)
- Supervised Machine Learning (Classification & Regression)
- Unsupervised Learning (Clustering, PCA, t-SNE)

Results are organized with expanders and columns for a clean look.
Graphs are styled using Seaborn’s whitegrid, and additional supervised
algorithms (SVM, Gradient Boosting) have been added.

Documentation and inline comments explain each section.

Author: Your Name
Date: YYYY-MM-DD

Instructions:
1. Install the required packages.
2. Run this app using "streamlit run app.py"
3. For deployment, refer to the README documentation in the GitHub repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set Seaborn style for better aesthetics
sns.set_style("whitegrid")

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Supervised Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Unsupervised Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------------------------------
# Sidebar option to display the code behind each operation
show_code = st.sidebar.checkbox("Show operation code", value=False)

def show_operation_code(code_str, language="python"):
    """Displays the code snippet if the checkbox is enabled."""
    if show_code:
        st.code(code_str, language=language)

# -------------------------------------------------------------------------
# Function to load a sample dataset
def load_sample_dataset(name):
    try:
        if name == "Iris":
            data = datasets.load_iris()
        elif name == "Wine":
            data = datasets.load_wine()
        else:  # Breast Cancer
            data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    except Exception as e:
        st.error("Error loading sample dataset: " + str(e))
        st.stop()

# -------------------------------------------------------------------------
# App Title & Documentation
st.title("Extended Supervised & Unsupervised Learning GUI")
st.markdown("""
This app allows you to perform a variety of operations including:
- **Exploratory Data Analysis (EDA)**
- **Supervised Machine Learning** (Classification & Regression)
- **Unsupervised Learning** (Clustering, PCA, t-SNE)

Use the sidebar to:
- Upload your CSV file or choose a sample dataset.
- Select the desired mode and operation.

Results are organized for clarity, and code snippets are available for each operation.
""")

# -------------------------------------------------------------------------
# Data Source Selection
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom CSV loaded.")
        data_source = "Custom CSV"
    except Exception as e:
        st.error("Error loading CSV: " + str(e))
        st.stop()
else:
    sample_dataset = st.sidebar.selectbox("Select Sample Dataset", ("Iris", "Wine", "Breast Cancer"))
    df = load_sample_dataset(sample_dataset)
    data_source = f"Sample Dataset: {sample_dataset}"
st.markdown(f"### Data Source: {data_source}")

if df.empty:
    st.error("The loaded dataset is empty. Please check your CSV file or sample selection.")
    st.stop()

# -------------------------------------------------------------------------
# Main Mode Selection
mode = st.sidebar.selectbox("Select Mode", ["Exploratory Data Analysis", "Machine Learning", "Unsupervised Learning"])

# =========================================================================
#                          EXPLORATORY DATA ANALYSIS
# =========================================================================
if mode == "Exploratory Data Analysis":
    eda_operation = st.sidebar.selectbox("Select EDA Operation", 
        ["Dataset Summary", "Data Types", "Histogram", "Box Plot", "Violin Plot",
         "Correlation Matrix", "Scatter Plot", "Pair Plot", "Missing Value Analysis",
         "Missing Value Heatmap", "Value Counts", "Distribution Plot", "Outlier Detection",
         "Skewness & Kurtosis", "Pivot Table"])
    
    st.header("Exploratory Data Analysis")
    
    # Use expanders to organize output sections
    if eda_operation == "Dataset Summary":
        with st.expander("Dataset Summary and Info"):
            try:
                st.subheader("Descriptive Statistics")
                st.write(df.describe())
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                code_summary = """
# Display descriptive statistics and DataFrame info
st.write(df.describe())
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())
"""
                show_operation_code(code_summary)
            except Exception as e:
                st.error("Error in Dataset Summary: " + str(e))
    
    elif eda_operation == "Data Types":
        with st.expander("Data Types"):
            try:
                st.subheader("Column Data Types")
                st.write(df.dtypes)
                code_dtype = "st.write(df.dtypes)"
                show_operation_code(code_dtype)
            except Exception as e:
                st.error("Error displaying Data Types: " + str(e))
    
    elif eda_operation == "Histogram":
        with st.expander("Histogram"):
            try:
                st.subheader("Histogram")
                hist_column = st.selectbox("Select column for histogram", df.columns)
                fig, ax = plt.subplots()
                ax.hist(df[hist_column].dropna(), bins=20, color='steelblue', edgecolor='black')
                ax.set_title(f"Histogram of {hist_column}")
                ax.set_xlabel(hist_column)
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                code_hist = f"""
fig, ax = plt.subplots()
ax.hist(df['{hist_column}'].dropna(), bins=20, color='steelblue', edgecolor='black')
ax.set_title("Histogram of {hist_column}")
ax.set_xlabel("{hist_column}")
ax.set_ylabel("Frequency")
st.pyplot(fig)
"""
                show_operation_code(code_hist)
            except Exception as e:
                st.error("Error in Histogram operation: " + str(e))
    
    elif eda_operation == "Box Plot":
        with st.expander("Box Plot"):
            try:
                st.subheader("Box Plot")
                box_column = st.selectbox("Select column for box plot", df.columns)
                fig, ax = plt.subplots()
                sns.boxplot(x=df[box_column], ax=ax, color='lightgreen')
                ax.set_title(f"Box Plot of {box_column}")
                st.pyplot(fig)
                code_box = f"""
fig, ax = plt.subplots()
sns.boxplot(x=df['{box_column}'], ax=ax, color='lightgreen')
ax.set_title("Box Plot of {box_column}")
st.pyplot(fig)
"""
                show_operation_code(code_box)
            except Exception as e:
                st.error("Error in Box Plot operation: " + str(e))
    
    elif eda_operation == "Violin Plot":
        with st.expander("Violin Plot"):
            try:
                st.subheader("Violin Plot")
                vio_column = st.selectbox("Select numeric column for violin plot", df.select_dtypes(include=np.number).columns)
                fig, ax = plt.subplots()
                sns.violinplot(y=df[vio_column], ax=ax, palette="muted")
                ax.set_title(f"Violin Plot of {vio_column}")
                st.pyplot(fig)
                code_vio = f"""
fig, ax = plt.subplots()
sns.violinplot(y=df['{vio_column}'], ax=ax, palette="muted")
ax.set_title("Violin Plot of {vio_column}")
st.pyplot(fig)
"""
                show_operation_code(code_vio)
            except Exception as e:
                st.error("Error in Violin Plot operation: " + str(e))
    
    elif eda_operation == "Correlation Matrix":
        with st.expander("Correlation Matrix"):
            try:
                st.subheader("Correlation Matrix")
                fig, ax = plt.subplots()
                corr = df.corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)
                code_corr = """
corr = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
st.pyplot(fig)
"""
                show_operation_code(code_corr)
            except Exception as e:
                st.error("Error in Correlation Matrix operation: " + str(e))
    
    elif eda_operation == "Scatter Plot":
        with st.expander("Scatter Plot"):
            try:
                st.subheader("Scatter Plot")
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) >= 2:
                    scatter_x = st.selectbox("X-axis", numeric_cols, index=0)
                    scatter_y = st.selectbox("Y-axis", numeric_cols, index=1)
                    fig, ax = plt.subplots()
                    ax.scatter(df[scatter_x], df[scatter_y], alpha=0.7, color='darkorange')
                    ax.set_xlabel(scatter_x)
                    ax.set_ylabel(scatter_y)
                    ax.set_title(f"Scatter Plot: {scatter_x} vs {scatter_y}")
                    st.pyplot(fig)
                    code_scatter = f"""
fig, ax = plt.subplots()
ax.scatter(df['{scatter_x}'], df['{scatter_y}'], alpha=0.7, color='darkorange')
ax.set_xlabel("{scatter_x}")
ax.set_ylabel("{scatter_y}")
ax.set_title("Scatter Plot: {scatter_x} vs {scatter_y}")
st.pyplot(fig)
"""
                    show_operation_code(code_scatter)
                else:
                    st.warning("Not enough numeric columns for scatter plot.")
            except Exception as e:
                st.error("Error in Scatter Plot operation: " + str(e))
    
    elif eda_operation == "Pair Plot":
        with st.expander("Pair Plot"):
            try:
                st.subheader("Pair Plot")
                fig = sns.pairplot(df.select_dtypes(include=np.number))
                st.pyplot(fig)
                code_pair = """
fig = sns.pairplot(df.select_dtypes(include=np.number))
st.pyplot(fig)
"""
                show_operation_code(code_pair)
            except Exception as e:
                st.error("Error in Pair Plot operation: " + str(e))
    
    elif eda_operation == "Missing Value Analysis":
        with st.expander("Missing Value Analysis"):
            try:
                st.subheader("Missing Value Analysis")
                missing = df.isnull().sum()
                st.write(missing)
                st.bar_chart(missing)
                code_missing = """
missing = df.isnull().sum()
st.write(missing)
st.bar_chart(missing)
"""
                show_operation_code(code_missing)
            except Exception as e:
                st.error("Error in Missing Value Analysis: " + str(e))
    
    elif eda_operation == "Missing Value Heatmap":
        with st.expander("Missing Value Heatmap"):
            try:
                st.subheader("Missing Value Heatmap")
                fig, ax = plt.subplots()
                sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
                ax.set_title("Missing Value Heatmap")
                st.pyplot(fig)
                code_missheat = """
fig, ax = plt.subplots()
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
ax.set_title("Missing Value Heatmap")
st.pyplot(fig)
"""
                show_operation_code(code_missheat)
            except Exception as e:
                st.error("Error in Missing Value Heatmap operation: " + str(e))
    
    elif eda_operation == "Value Counts":
        with st.expander("Value Counts"):
            try:
                st.subheader("Value Counts")
                col = st.selectbox("Select column for value counts", df.columns)
                counts = df[col].value_counts()
                st.write(counts)
                st.bar_chart(counts)
                code_counts = f"""
counts = df['{col}'].value_counts()
st.write(counts)
st.bar_chart(counts)
"""
                show_operation_code(code_counts)
            except Exception as e:
                st.error("Error in Value Counts operation: " + str(e))
    
    elif eda_operation == "Distribution Plot":
        with st.expander("Distribution Plot"):
            try:
                st.subheader("Distribution Plot")
                dist_col = st.selectbox("Select column for distribution plot", df.select_dtypes(include=np.number).columns)
                fig, ax = plt.subplots()
                sns.histplot(df[dist_col].dropna(), kde=True, ax=ax, color='mediumseagreen')
                ax.set_title(f"Distribution of {dist_col}")
                st.pyplot(fig)
                code_dist = f"""
fig, ax = plt.subplots()
sns.histplot(df['{dist_col}'].dropna(), kde=True, ax=ax, color='mediumseagreen')
ax.set_title("Distribution of {dist_col}")
st.pyplot(fig)
"""
                show_operation_code(code_dist)
            except Exception as e:
                st.error("Error in Distribution Plot operation: " + str(e))
    
    elif eda_operation == "Outlier Detection":
        with st.expander("Outlier Detection"):
            try:
                st.subheader("Outlier Detection using IQR")
                num_col = st.selectbox("Select numeric column for outlier detection", df.select_dtypes(include=np.number).columns)
                Q1 = df[num_col].quantile(0.25)
                Q3 = df[num_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[num_col] < lower_bound) | (df[num_col] > upper_bound)]
                st.write(f"Number of outliers in {num_col}: {len(outliers)}")
                st.write(outliers)
                code_outliers = f"""
Q1 = df['{num_col}'].quantile(0.25)
Q3 = df['{num_col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['{num_col}'] < lower_bound) | (df['{num_col}'] > upper_bound)]
st.write("Number of outliers in {num_col}: " + str(len(outliers)))
st.write(outliers)
"""
                show_operation_code(code_outliers)
            except Exception as e:
                st.error("Error in Outlier Detection: " + str(e))
    
    elif eda_operation == "Skewness & Kurtosis":
        with st.expander("Skewness & Kurtosis"):
            try:
                st.subheader("Skewness & Kurtosis")
                numeric_cols = df.select_dtypes(include=np.number).columns
                stats_df = pd.DataFrame({
                    "Skewness": df[numeric_cols].skew(),
                    "Kurtosis": df[numeric_cols].kurtosis()
                })
                st.write(stats_df)
                code_skew = """
numeric_cols = df.select_dtypes(include=np.number).columns
stats_df = pd.DataFrame({
    "Skewness": df[numeric_cols].skew(),
    "Kurtosis": df[numeric_cols].kurtosis()
})
st.write(stats_df)
"""
                show_operation_code(code_skew)
            except Exception as e:
                st.error("Error in Skewness & Kurtosis: " + str(e))
    
    elif eda_operation == "Pivot Table":
        with st.expander("Pivot Table"):
            try:
                st.subheader("Pivot Table")
                cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not cat_cols:
                    st.warning("No categorical columns found for pivot table. Please use a CSV with categorical data.")
                else:
                    pivot_cat = st.selectbox("Select categorical column", cat_cols)
                    if num_cols:
                        pivot_num = st.selectbox("Select numeric column", num_cols)
                        pivot = pd.pivot_table(df, index=pivot_cat, values=pivot_num, aggfunc=np.mean)
                        st.write(pivot)
                        code_pivot = f"""
pivot = pd.pivot_table(df, index='{pivot_cat}', values='{pivot_num}', aggfunc=np.mean)
st.write(pivot)
"""
                        show_operation_code(code_pivot)
                    else:
                        st.warning("No numeric columns available for pivot operation.")
            except Exception as e:
                st.error("Error in Pivot Table operation: " + str(e))

# =========================================================================
#                     MACHINE LEARNING OPERATIONS
# =========================================================================
elif mode == "Machine Learning":
    ml_operation = st.sidebar.selectbox("Select ML Operation", ["Classification", "Regression", "Feature Selection & CV"])
    st.header("Machine Learning Operations")
    
    # Select target and features
    target = st.sidebar.selectbox("Select Target Column", df.columns)
    feature_cols = st.multiselect("Select Feature Columns", options=[col for col in df.columns if col != target])
    if not feature_cols:
        feature_cols = [col for col in df.columns if col != target]
    
    # Data splitting with error handling
    try:
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20, step=5)
        X = df[feature_cols]
        y = df[target]
        if X.empty or y.empty:
            st.error("Selected features or target is empty. Please recheck your selections.")
            st.stop()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        st.error("Error during train-test split or scaling: " + str(e))
        st.stop()
    
    # ------------------ Classification ------------------
    if ml_operation == "Classification":
        with st.expander("Classification Operations"):
            try:
                # Check that target is categorical
                if y.nunique() > 10:
                    st.warning(
                        "The selected target column appears to be continuous (more than 10 unique values). "
                        "Classification models require discrete class labels. Please select a different column or use Regression mode."
                    )
                    st.stop()
                st.subheader("Classification")
                classifier = st.sidebar.selectbox("Select Classification Model", 
                    ["Logistic Regression", "K-Nearest Neighbors", "Random Forest", "Support Vector Machine"])
                if classifier == "Logistic Regression":
                    C = st.sidebar.number_input("C (Inverse regularization strength)", 0.01, 10.0, value=1.0, step=0.01)
                    model = LogisticRegression(C=C, max_iter=1000)
                    code_model = f"model = LogisticRegression(C={C}, max_iter=1000)"
                elif classifier == "K-Nearest Neighbors":
                    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 15, value=5)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    code_model = f"model = KNeighborsClassifier(n_neighbors={n_neighbors})"
                elif classifier == "Random Forest":
                    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, value=100, step=10)
                    max_depth = st.sidebar.slider("Max Depth", 2, 15, value=5)
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    code_model = f"model = RandomForestClassifier(n_estimators={n_estimators}, max_depth={max_depth}, random_state=42)"
                elif classifier == "Support Vector Machine":
                    C = st.sidebar.number_input("C (Penalty Parameter)", 0.01, 10.0, value=1.0, step=0.01)
                    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVC(C=C, kernel=kernel, probability=True)
                    code_model = f"model = SVC(C={C}, kernel='{kernel}', probability=True)"
                st.write("### Training the Classification Model...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.2f}")
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                code_classification = code_model + """
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
st.write("Accuracy:", acc)
st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
st.write("Classification Report:", classification_report(y_test, y_pred))
"""
                show_operation_code(code_classification)
            except Exception as e:
                st.error("Error during Classification: " + str(e))
    
    # ------------------ Regression ------------------
    elif ml_operation == "Regression":
        with st.expander("Regression Operations"):
            try:
                st.subheader("Regression")
                regressor = st.sidebar.selectbox("Select Regression Model", 
                    ["Linear Regression", "K-Nearest Neighbors", "Random Forest", "Gradient Boosting"])
                if regressor == "Linear Regression":
                    model = LinearRegression()
                    code_model = "model = LinearRegression()"
                elif regressor == "K-Nearest Neighbors":
                    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 15, value=5)
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                    code_model = f"model = KNeighborsRegressor(n_neighbors={n_neighbors})"
                elif regressor == "Random Forest":
                    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, value=100, step=10)
                    max_depth = st.sidebar.slider("Max Depth", 2, 15, value=5)
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                    code_model = f"model = RandomForestRegressor(n_estimators={n_estimators}, max_depth={max_depth}, random_state=42)"
                elif regressor == "Gradient Boosting":
                    n_estimators = st.sidebar.slider("Number of Trees", 10, 300, value=100, step=10)
                    learning_rate = st.sidebar.number_input("Learning Rate", 0.01, 1.0, value=0.1, step=0.01)
                    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                    code_model = f"model = GradientBoostingRegressor(n_estimators={n_estimators}, learning_rate={learning_rate}, random_state=42)"
                st.write("### Training the Regression Model...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")
                code_regression = code_model + """
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write("Mean Squared Error:", mse)
st.write("R² Score:", r2)
"""
                show_operation_code(code_regression)
            except Exception as e:
                st.error("Error during Regression: " + str(e))
    
    # ------------------ Feature Selection & Cross-Validation ------------------
    elif ml_operation == "Feature Selection & CV":
        with st.expander("Feature Selection & Cross-Validation"):
            try:
                st.subheader("Feature Selection & Cross-Validation")
                st.write("Performing 5-fold cross-validation using Logistic Regression.")
                model = LogisticRegression(max_iter=1000)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                st.write("Cross-validation scores:", scores)
                st.write("Mean CV score:", np.mean(scores))
                code_cv = """
from sklearn.model_selection import cross_val_score
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
st.write("Cross-validation scores:", scores)
st.write("Mean CV score:", np.mean(scores))
"""
                show_operation_code(code_cv)
            except Exception as e:
                st.error("Error during Feature Selection & CV: " + str(e))

# =========================================================================
#                      UNSUPERVISED LEARNING OPERATIONS
# =========================================================================
elif mode == "Unsupervised Learning":
    unsup_operation = st.sidebar.selectbox("Select Unsupervised Operation", ["KMeans Clustering", "PCA", "t-SNE"])
    st.header("Unsupervised Learning Operations")
    
    # Use only numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.error("No numeric data available for unsupervised operations.")
        st.stop()
    
    # Standardize numeric data
    scaler_unsup = StandardScaler()
    data_scaled = scaler_unsup.fit_transform(numeric_df)
    
    if unsup_operation == "KMeans Clustering":
        with st.expander("KMeans Clustering"):
            try:
                st.subheader("KMeans Clustering")
                n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, value=3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(data_scaled)
                numeric_df['Cluster'] = clusters
                st.write("Cluster assignments:")
                st.write(numeric_df['Cluster'].value_counts())
                # PCA projection for visualization
                pca_temp = PCA(n_components=2)
                pcs = pca_temp.fit_transform(data_scaled)
                fig, ax = plt.subplots()
                scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=clusters, cmap="viridis", alpha=0.7)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("KMeans Clustering (PCA Projection)")
                st.pyplot(fig)
                code_kmeans = f"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters={n_clusters}, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(data_scaled)
fig, ax = plt.subplots()
scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=clusters, cmap="viridis", alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("KMeans Clustering (PCA Projection)")
st.pyplot(fig)
"""
                show_operation_code(code_kmeans)
            except Exception as e:
                st.error("Error during KMeans Clustering: " + str(e))
    
    elif unsup_operation == "PCA":
        with st.expander("PCA"):
            try:
                st.subheader("Principal Component Analysis (PCA)")
                n_components = st.sidebar.slider("Number of Components", 2, min(10, numeric_df.shape[1]), value=2)
                pca = PCA(n_components=n_components)
                pcs = pca.fit_transform(data_scaled)
                st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
                fig, ax = plt.subplots()
                ax.plot(range(1, n_components+1), pca.explained_variance_ratio_, marker='o', color='teal')
                ax.set_xlabel("Principal Component")
                ax.set_ylabel("Explained Variance Ratio")
                ax.set_title("PCA Explained Variance")
                st.pyplot(fig)
                if n_components >= 2:
                    fig2, ax2 = plt.subplots()
                    ax2.scatter(pcs[:, 0], pcs[:, 1], alpha=0.7, color='orange')
                    ax2.set_xlabel("PC1")
                    ax2.set_ylabel("PC2")
                    ax2.set_title("PCA Projection (First 2 Components)")
                    st.pyplot(fig2)
                code_pca = f"""
from sklearn.decomposition import PCA
pca = PCA(n_components={n_components})
pcs = pca.fit_transform(data_scaled)
st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
fig, ax = plt.subplots()
ax.plot(range(1, {n_components}+1), pca.explained_variance_ratio_, marker='o', color='teal')
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Explained Variance")
st.pyplot(fig)
if {n_components} >= 2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(pcs[:, 0], pcs[:, 1], alpha=0.7, color='orange')
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA Projection (First 2 Components)")
    st.pyplot(fig2)
"""
                show_operation_code(code_pca)
            except Exception as e:
                st.error("Error during PCA: " + str(e))
    
    elif unsup_operation == "t-SNE":
        with st.expander("t-SNE"):
            try:
                st.subheader("t-SNE")
                perplexity = st.sidebar.slider("Perplexity", 5, 50, value=30)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_result = tsne.fit_transform(data_scaled)
                fig, ax = plt.subplots()
                ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, color='magenta')
                ax.set_xlabel("t-SNE Dimension 1")
                ax.set_ylabel("t-SNE Dimension 2")
                ax.set_title("t-SNE Projection")
                st.pyplot(fig)
                code_tsne = f"""
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity={perplexity}, random_state=42)
tsne_result = tsne.fit_transform(data_scaled)
fig, ax = plt.subplots()
ax.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7, color='magenta')
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("t-SNE Projection")
st.pyplot(fig)
"""
                show_operation_code(code_tsne)
            except Exception as e:
                st.error("Error during t-SNE: " + str(e))

st.markdown("""
---
**Documentation:**
- This app is designed as a comprehensive tool for data analysis and model experimentation.
- For more information, please refer to the [README](README.md) in this repository.
- Feel free to extend the operations and add more algorithms as needed.
""")
