import streamlit as st
import pandas as pd

# Additional styling
st.markdown(
    """
    <style>
        body {
            background-color: #E6E6FA;
        }
        h1 {
            color: #4B0082;
        }
        .stMarkdown blockquote {
            border-left: 3px solid #4B0082;
            padding-left: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to create LinkedIn button
def linkedin_button(name, url):
    return (
        f'<a href="{url}" target="_blank"><img src="https://img.shields.io/badge/{name}-0077B5?style=for-the-badge'
        f'&logo=linkedin&logoColor=white" /></a>'
    )


# Home Section
def home():
    st.title("Analysis of Excess Stock Returns Prediction Methods")
    st.subheader("Purpose of the Website")
    st.write(
        """This website is designed to showcase our research project: an exhaustive analysis of the methods to 
        forecast the excess stock returns. It includes details on the data we used, how we processed them, 
        the models we explored and their performances. For any inquiries, please contact: PROJECT EMAIL TO ADD."""
    )
    st.subheader("Some Results")
    st.write("Here you can include some key findings or results.")

    # Adding the table with icons
    st.markdown(
        """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid black;
                text-align: center;
                padding: 8px;
            }
        </style>
        <table>
            <thead>
                <tr>
                    <th></th>
                    <th>PCR</th>
                    <th>PLS</th>
                    <th>Random Forest</th>
                    <th>Boosted Trees</th>
                    <th>Neural Networks</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>📊 OOS R^2</td>
                    <td>Value1</td>
                    <td>Value2</td>
                    <td>Value3</td>
                    <td>Value4</td>
                    <td>Value5</td>
                </tr>
                <tr>
                    <td>📈 Sharpe Ratios</td>
                    <td>Value1</td>
                    <td>Value2</td>
                    <td>Value3</td>
                    <td>Value4</td>
                    <td>Value5</td>
                </tr>
                <tr>
                    <td>🔍 Most Important Variables</td>
                    <td>Value1</td>
                    <td>Value2</td>
                    <td>Value3</td>
                    <td>Value4</td>
                    <td>Value5</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Framework")

    st.latex(
        r"""
    \forall t\in \left \{ 1957-03,1957-04, ..., 2016-12 \right \}, 
    \forall j\in \left \{ \text{company}_1 ,\text{company}_2, ..., \text{company}_{26709} \right \},
    """
    )

    st.write("We have the following prediction equation:")

    st.latex(
        r"""
    \hat{y}_{j,t} = f(X^1_{j,t}, X^2_{j,t}, ..., X^{920}_{j,t})
    """
    )

    st.write("With")
    st.latex(
        """
    X^i_{j,t}, \ f
    """
    )
    st.write(
        "respectively representing the signal i for company j at time t and the predictor function."
    )

    st.write(
        "Except for the 74 industry dummy variables, the signals are constructed the following way:"
    )

    st.latex(
        r"""
    \forall i\in  \left [ 1, 2, ..., 920 - 74 \right ], 
    \exists k \in \left [ 1, 2, ..., 94 \right ], \exists l \in \left [ 1, 2, ..., 9 \right ]
    """
    )

    st.latex(
        r"""
    X^i_{j,t} = \text{Characteristic}^k_{j,t} \times \text{MacroPredictor}^l_{t}
    """
    )

    st.write("Which is why we have \( 94 \times 9 + 74 = 920 \) signals.")

    st.subheader("Authors")

    # LinkedIn buttons displayed horizontally
    st.write(
        f'<div style="display: flex; flex-direction: row; justify-content: space-between; margin: auto;">'
        f'{linkedin_button("Harshita Garg", "https://www.linkedin.com/in/garg-harshita/")}'
        f'{linkedin_button("Evelyn Piao", "https://www.linkedin.com/in/yimeng-evelyn-piao/")}'
        f'{linkedin_button("Qi Zhang", "https://www.linkedin.com/in/jenny-qi-zhang/")}'
        f'{linkedin_button("Sofiane Bessaoud", "https://www.linkedin.com/in/sofiane-bessaoud/")}'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Thank You Note Styled
    st.subheader("Acknowledgments")
    st.markdown(
        """
        > **Special Thanks to:**
        >
        > _We would like to extend our heartfelt gratitude to **Ali Kakhbod** for his invaluable supervision and guidance,
        > as well as to **Andrew Lazzeri** for his insightful advice._
        """,
        unsafe_allow_html=True,
    )

    # Paper Reference Styled
    st.subheader("Inspiration")
    st.markdown(
        """
        > **Motivated By:**
        >
        > _This project was greatly inspired by the paper **"Empirical Asset Pricing via Machine Learning"** authored by 
        > **Shihao Gu**, **Bryan Kelly**, and **Dacheng Xiu**._
        """,
        unsafe_allow_html=True,
    )


# Performance Section
def performances():
    st.title("Model Performances")
    st.subheader("Performance Metrics")
    st.write(
        """
    Detailed performance metrics for each model, including accuracy, precision, recall, F1-score, etc.
    """
    )


# Modeling Section
def modeling():
    st.title("Modeling")
    st.subheader("Models Used")
    st.write(
        """
    Explanation of the models used in this project, including algorithms, hyperparameters, and any other relevant details.
    """
    )


# Data Section
# Data Section
def data():
    st.title("Data Overview")
    st.subheader("900 signals and monthly returns from 1957 to 2016")
    st.write(
        "We obtained from WRDS the monthly total equity returns annually annualized from CRSP for all firms listed "
        "in the NYSE, AMEX, NASDAQ and ARCA. The data starts when the S&P500 was created in March 1957 "
        "and ends in December 2016. "
    )
    st.write(
        """
    We built a predictors dataset composed of 900 signals for every company by computing the interactions between 94 company-level 
    characteristics and 8 macro economic-factors (+ 1 constant to keep the raw characteristic), and 74 industry dummies. 
    """
    )
    st.write(
        """
    We obtained the characteristics dataset here: https://dachxiu.chicagobooth.edu/ and the macroeconomic predictors 
    here: https://sites.google.com/view/agoyal145
    """
    )
    st.subheader("Data Engineering")
    st.write(
        """
    Explanation of the data engineering process, including cleaning, transformation, and feature engineering.
    """
    )

    st.subheader("Macroeconomic factors")
    macro_df = pd.read_csv("2000_macro_factors_preprocessed.csv")
    st.markdown(
        """
        After downloading the macroeconomic predictors dataset, we implemented the feature engineering explained in this paper: 
        [<span style='color: blue; text-decoration: underline;'>A Comprehensive Look at The Empirical Performance of Equity Premium Prediction</span>](https://drive.google.com/file/d/1uvjBJ9D09T0_sp7kQppWpD-xelJ0KQhc/view?pli=1)
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(macro_df)

    st.subheader("Company Characteristics and Returns")
    st.markdown(
        "An extract of the dataset of the IBM company for years 2000 and 2001 is available bellow."
        "After obtaining the characteristics data set, we noticed that for all companies and dates, "
        "realestate and secured features did not have data. We decided to drop them. We filled the NaN values "
        "with the median from the training dataset"
    )
    ibm_df = pd.read_csv("ibm_2000_2002_charac_returns.csv")
    st.dataframe(ibm_df)


# Sidebar Navigation with icons
st.sidebar.title("🌐 Navigation")
section = st.sidebar.radio(
    "Go to", ["🏠 Home", "📊 Performances", "🛠 Modeling", "📦 Data"]
)

if section == "🏠 Home":
    home()
elif section == "📊 Performances":
    performances()
elif section == "🛠 Modeling":
    modeling()
elif section == "📦 Data":
    data()
