import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

#look for more information here https://docs.streamlit.io/library/cheatsheet

#adding title
st.title("Title Here")

#adding discription to your website
st.text('Description')

st.write()

df = pd.read_csv("report_2018-2019.csv")

#Chloe
st.header("Introduction")
st.text('This week, we were given data about the happiness levels of different countries. We used this data to identify what factors affect happiness and how they do so.')


#Does a higher happiness index score lead to higher life expectancy?

#Scatter Plot with seaborn, matplotlib

#What countries have the highest and lowest happiness index?

#Bar chart

#Isaiah
#Does generosity affect Happiness Score of a country?
#Scatter Plot hue = country
import seaborn as sns     #
sns.set_theme()

sns.scatterplot()
    data=df,
    x="Generosity", y="Score")  # need to study more

#Taylor
#Is Perception of Corruption correlated to happiness?
st.header("Hypothesis: Is Perception of Corruption correlated to happiness?")


df.head(10)
import seaborn as sns
sns.set_theme()


sns.scatterplot(
    data=df,
    x="Perceptions of corruption", y="Score"
)

#Scatter Plot
#Line

#Marcus
#Does Freedom to make Choices to Happiness correlate to happines?
sns.scatterplot(
    data=df,
    x="Social support", y="Score")
#Does Social Support correlate to happines?
sns.scatterplot(
    data=df,
   x="Freedom to make life choices", y="Score")

#Matthew
#Does high/low gdp per capita correlate to happines?

import pandas as pd
st.header("Hypothesis: Does a higher GDP per capita lead to a higher level of happiness?")
#Cleaning Data
df.dropna(inplace = True)
df.drop_duplicates(inplace = True)
#Scatter plot, hue = country
import seaborn as sns
sns.scatterplot(data = df, x="GDP per capita", y="Score")
#Can we predict Happiness score using GDP Per Capita?
from seaborn.matrix import heatmap
sns.heatmap(df.corr())

#Linear Regression Plot

#Conclusion
