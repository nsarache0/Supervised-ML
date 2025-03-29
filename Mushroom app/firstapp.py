import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
# Set page config to make it more personalized and dark theme
st.set_page_config(page_title="Mushroom Classification", page_icon="🍄", layout="wide")


def main():
  st.title("Binary classification app")
if __name__ == "__main__":
    main()
st.sidebar.title("Binary Classification web app")


@st.cache_data
def load_data():
  data = pd.read_csv("C:\\Users\\nico-\\Downloads\\mushrooms.csv")
  label = LabelEncoder()
  for col in data.columns:
     data[col] = label.fit_transform(data[col])
  return data


@st.cache_data

def split(df):
   y = df.type
   x = df.drop(columns = ['type'])
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 0)
   return x_train, x_test, y_train, y_test


def plot_metrics(model, X_test, y_test, class_names, metric_list):
    if "Confusion Matrix" in metric_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=class_names, ax=ax, cmap="Blues")
        st.pyplot(fig)  # Streamlit requires passing the figure explicitly

    if "ROC Curve" in metric_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

    if "Precision-Recall Curve" in metric_list:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

df = load_data()
x_train, x_test, y_train, y_test = split(df)
class_names = ["Edibles", "Poisonous"]

st.sidebar.markdown("Find if your mushroom is edible or posionous 🍄")
if st.sidebar.checkbox("Show raw data", False):
   st.subheader("Mushroom data set - Classification")
   st.write(df)

st.sidebar.subheader("Choose a Classification Model")
classifier = st.sidebar.selectbox("Algorithm:", ["Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Naive Bayes", "AdaBoost"])

##svm
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Select Hyperparameters for SVM")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel type", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar.radio("Kernel coefficient", ("scale", "auto"))
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="Classify"):  # Correct indentation
        st.subheader("Support Vector Machine (SVM) Results:")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)  # Assuming you have defined x_train and y_train
        accuracy = model.score(x_test, y_test)  # Assuming you have defined x_test and y_test
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))  # Rounding precision
        st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 2))  # Rounding recall  # Assuming plot_metrics is a defined function  # Assuming you defined class_names
        plot_metrics(model, x_test, y_test, class_names, metrics)  

##logistic regression:
if classifier == "Logistic Regression":
    st.sidebar.subheader("Select Hyperparameters for LR")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    max_iter = st.sidebar.slider("Maximum number of iterations", 100,500, key = "max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="Classify"):  
        st.subheader("Logistic regression Results:")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)  # Assuming you have defined x_train and y_train
        accuracy = model.score(x_test, y_test)  # Assuming you have defined x_test and y_test
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))  # Rounding precision
        st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 2))  # Rounding recall  # Assuming plot_metrics is a defined function  # Assuming you defined class_names
        plot_metrics(model, x_test, y_test, class_names, metrics) 

##rf
if classifier == "Random Forest":
    st.sidebar.subheader("Select Hyperparameters for RF")
    n_estimators = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 20, key="n_estimators")
    max_depth = st.sidebar.number_input("Maximum depth of trees", 1, 30, step = 2, key = "max_depth")

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="Classify"):  
        st.subheader("Random Forest results:")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= False, n_jobs = -1)
        model.fit(x_train, y_train)  # Assuming you have defined x_train and y_train
        accuracy = model.score(x_test, y_test)  # Assuming you have defined x_test and y_test
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))  # Rounding precision
        st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 2))  # Rounding recall  # Assuming plot_metrics is a defined function  # Assuming you defined class_names
        plot_metrics(model, x_test, y_test, class_names, metrics)

## Naive Bayes (MultinomialNB for categorical features)
if classifier == "Naive Bayes":
    st.sidebar.subheader("Select Hyperparameters for Naive Bayes")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="Classify"):
        st.subheader("Naive Bayes Results:")
        model = MultinomialNB()  # Using MultinomialNB for categorical features
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 2))
        plot_metrics(model, x_test, y_test, class_names, metrics)

## AdaBoost
if classifier == "AdaBoost":
    st.sidebar.subheader("Select Hyperparameters for AdaBoost")
    n_estimators = st.sidebar.number_input("Number of estimators", 50, 200, step=10, key="n_estimators")
    learning_rate = st.sidebar.number_input("Learning rate", 0.01, 1.0, step=0.01, key="learning_rate")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="Classify"):
        st.subheader("AdaBoost Results:")
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred, labels=class_names), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred, labels=class_names), 2))
        plot_metrics(model, x_test, y_test, class_names, metrics)
  
  
   


  


