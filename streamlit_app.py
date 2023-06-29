import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import io

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

#look for more information here https://docs.streamlit.io/library/cheatsheet

#adding title
st.title("Title Here")

#adding discription to your website
st.text('Description')

df = pd.read_csv("report_2018-2019.csv")

print(df.head(2))
