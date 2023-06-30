import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

#look for more information here https://docs.streamlit.io/library/cheatsheet

#adding title
st.title("Vivacious Vultures")

st.write(
  "Hello, I am Chloe and I am a rising senior from Florida. I took this course in order to explore data science and learn some coding."
)

st.write(
  "hi im Isaiah im gonna be a senior and I'm from Texas. i decided to take this course so i cam better understand coding so that i myself can use it"
)

st.write(
  "Hi, I'm Taylor and I am a soon to be sophmore.  I took this course because I wanted to learn more about data science and hopefully gain some skills."
)

st.write(
  "Hello, I'm Marcus Smissen, a soon to be junion. I decided to take this  week long course so I could learn more about python."
)

st.write(
  "Hello, I'm Matthew and I am a rising senior. I took this course to explore how data science can be used in applications of coding and I hope to work with more coding and data science."
)

df = pd.read_csv("report_2018-2019.csv")

#Chloe
st.header("Introduction")
st.write(
  "This week, we were given data about the happiness levels of different countries. Along with each country's happiness index, we were also given other pieces of information that may factor in how content a region is. We used this data to create hypothesises and identify what affects happiness and how they do so."
)

st.header(
  'Does a higher happiness index score lead to higher life expectancy?')

sns.set_theme()
sns.scatterplot(
  data=df,
  x="Score",
  y="Healthy life expectancy",
)

st.write(
  'The  evidence suggests that those who live in coutries with a higher happiness score are more likely to live longer. There is a correlation between the two. As observed, the life expectancy increases as the score increases.'
)

st.pyplot()
#Scatter Plot

st.header('What countries have the highest and lowest happiness index?')

dfw = df[df["Year"] == 2018]
sorted_dfw = dfw.sort_values('Score', ascending=False)
fig = px.bar(sorted_dfw.head(10),
             x='Country or region',
             y='Score',
             title='High Scoring Countries')
st.plotly_chart(fig)

dfw = df[df["Year"] == 2018]
sorted_dfw = dfw.sort_values('Score', ascending=True)
fig = px.bar(sorted_dfw.head(10),
             x='Country or region',
             y='Score',
             title='Low Scoring Countries')
st.plotly_chart(fig)

st.write(
  'As the bar graphs shows, the countries with the highest happiness index are Finland, Norway and Denmark. Finland has the highest score of 7.632. On the other hand the countries with the lowest scores are Burundi, Central African Republic and South Sudan. Burundi has the lowest score of 2.905, a little over a third of the score Finland received.'
)
#Bar Chart

#Isaiah
#Does generosity affect Happiness Score of a country?
#Scatter Plot hue = country
import seaborn as sns

sns.set_theme()

st.header("hypothesis: does generosity affect the happiness score of a country")
import seaborn as sns     #
sns.set_theme()

url_dataframe = 'https://github.com/Shaif95/AI_camp/raw/main/report_2018-2019.csv'


sns.scatterplot(
    data=df,
    x="Generosity", y="Score")

st.write( "The scatter plot shows that whether there are low or high rates of generosity its effects on happiness are minimal ")
st.pyplot()

#Taylor
#Is Perception of Corruption correlated to happiness?
import pandas as pd

st.header("Hypothesis: Is Perception of Corruption correlated to happiness?")

import seaborn as sns

sns.set_theme()

sns.scatterplot(data=df, x="Perceptions of corruption", y="Score")

import matplotlib.pyplot as plt

x_values = [0.1, 0.2, 0.3, 0.4]
y_values = [2, 4, 6, 8]
plt.scatter(x_values, y_values)
line_slope = 20
line_intercept = 0
line_values = [line_slope * x + line_intercept for x in x_values]
plt.plot(x_values, line_values, color='red')
plt.show()

st.pyplot()


st.write(
  "Higher perceptions of corruption are mostly correlated with a higher happiness index. Most countries with a happiness index from 7-8 have a perception of corruption of 0.3-0.4. These are most likely correlated beacuse the higher the perception of corruption is the less corrupt a country is. Most likely leading to the people of the country to be more content and happy."
)
#Scatter Plot
#Line

#Marcus
sns.set_theme()
#Does Social Support correlate to happiness?
sns.scatterplot(data=df, x="Social support", y="Score")

st.pyplot()

st.write(
  "The amount of social support a country has is likely correlated to to a higher level of happiness shown within the happiness index, as most countries with a 7 or above have higher values of social support (typically at values 1.4 or higher)."
)
#Does Freedom to make Choices to Happiness correlate to happiness?
sns.scatterplot(data=df, x="Freedom to make life choices", y="Score")

st.pyplot()

st.write(
  "The freedom to make life choices is likely correlated to to a higher level of happiness shown within the happiness index, as most countries with a 7 or above have higher values of claimed freedom to make life choices (typically at values 0.5 or higher)."
)

#Matthew
#Does high/low gdp per capita correlate to happines?
st.header(
  "Hypothesis: Does a higher GDP per capita lead to a higher level of happiness?"
)
#Cleaning Data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
#Scatter plot, hue = country
import seaborn as sns

sns.lineplot(data=df, x="GDP per capita", y="Score")

st.pyplot()

from seaborn.matrix import heatmap

df = df.dropna()

# Convert non-numeric columns to numeric types if necessary
df = df.astype(float)

fig56 =  sns.heatmap(df.corr())

st.pyplot(fig56)

st.write(
  "As seen in the heatmap, GDP and score have a high level of correlation. This means that countries who have a higher GDP per capita tend to also have a high level of happiness. However, while this does seem possible, this isnt always 100% true. The country with the highest GDP per capita (United States of America) does not have the hgihest level of happiness, meaning that GDP may not always lead to hgih levels of happiness. Nonetheless, as seen within the lineplot, there is generally a positive correlation between GDP per Capita and Score."
)
st.dataframe(df[df["Overall rank"] <= 5].head(20))
#Can we predict Happiness score using GDP Per Capita?
st.header("Can we predict a happiness score using GDP Per Capita")
#Linear Regression Plot
import statistics

length = len(df.index)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the necessary data and preprocessing steps already performed

X = df['GDP per capita'].to_numpy()
y = df['Score'].to_numpy()

length = len(X)
idx = np.arange(length)
np.random.shuffle(idx)
split_threshold = int(length * 0.8)
train_idx = idx[:split_threshold]
test_idx = idx[split_threshold:]
x_train, y_train = X[train_idx], y[train_idx]
x_test, y_test = X[test_idx], y[test_idx]
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Calculate the coefficients using numpy
X_train = np.concatenate((x_train, np.ones_like(x_train)), axis=1)
coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Extract the slope and intercept
slope = coefficients[0][0]
intercept = coefficients[1][0]

# Make predictions on the test data
y_hat = slope * x_test + intercept

# Plotting the data
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_test, y_hat, '--')
ax.scatter(x_test, y_test, c='red')
ax.set_xlabel('GDP per Capita', fontsize=20)
ax.set_ylabel('Happiness Score', fontsize=20)
ax.set_title('Happiness Score vs GDP per Capita')
ax.grid(True)

# Display the plot using Streamlit
st.pyplot(fig)

st.write(
  "As seen in the plot of linear regression an increase in GDP also typically leads to an increase in score. Still, we are able to see that whie Happiness Score and GDP per Capita might be related they are not exact. This means we might be able to make close prediction of what a country would look like with a certain GDP but it wouldnt be fully accurate. However, this does offer the possibility of a fairly accurate prediction."
)
#Conclusion
