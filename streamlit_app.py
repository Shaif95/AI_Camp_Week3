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
st.text('This week, we were given data about the happiness levels of ')


#Does a higher happiness index score lead to higher life expectancy?

#Scatter Plot with seaborn, matplotlib

#What countries have the highest and lowest happiness index?

#Bar chart

#Isaiah
#Does generosity affect Happiness Score of a country?
#Scatter Plot hue = country


#Taylor
#Is Perception of Corruption correlated to happiness?
st.header

import seaborn as sns
sns.set_theme()

url_dataframe = 'https://github.com/Shaif95/AI_camp/raw/main/report_2018-2019.csv'


sns.scatterplot(
    data=df,
    x="Perceptions of corruption", y="Score"
)

#Scatter Plot
#Line

#Marcus
#Does Freedom to make Choices to Happiness correlate to happines?
#Does Social Support correlate to happines?

#Matthew
#Does high/low gdp per capita correlate to happines?
url = 'https://github.com/Shaif95/AI_camp/raw/main/report_2018-2019.csv'

filename = wget.download(url)
import pandas as pd

df = pd.read_csv(filename)


#Scatter plot, hue = country

#Can we predict Happiness score using GDP Per Capita?
#Linear Regression Plot

#Conclusion
