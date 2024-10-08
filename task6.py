# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('disney_plus_titles.csv')

# Display the first few rows of the dataset
df.head()

# 1. Data Cleaning
# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Handling missing values - Dropping rows with missing values for simplicity
df_cleaned = df.dropna()

# Checking for outliers in release_year using boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='release_year', data=df_cleaned)
plt.title('Boxplot for Release Year')
plt.show()

# 2. Exploratory Data Analysis (EDA)
# Overview of the dataset after cleaning
print(df_cleaned.describe())

# Distribution of content types (Movies vs. TV Shows)
plt.figure(figsize=(8, 6))
sns.countplot(x='type', data=df_cleaned)
plt.title('Content Type Distribution (Movies vs. TV Shows)')
plt.xlabel('Type')
plt.ylabel('Number of Titles')
plt.show()

# Distribution of content ratings
plt.figure(figsize=(10, 6))
sns.countplot(x='rating', data=df_cleaned)
plt.xticks(rotation=45)
plt.title('Distribution of Content Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 3. Question Formulation & Answers

# Question 1: What is the distribution of content release over the years?
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['release_year'], bins=30, kde=False)
plt.title('Distribution of Content Release Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.show()

# Question 2: What are the most common genres in Disney Plus content?
# Split genres into individual items for analysis
df_cleaned['listed_in'] = df_cleaned['listed_in'].apply(lambda x: x.split(', '))
genres = df_cleaned['listed_in'].explode().value_counts()
plt.figure(figsize=(10, 6))
genres.head(10).plot(kind='bar')
plt.title('Top 10 Most Common Genres in Disney Plus')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

# Question 3: What is the distribution of content ratings?
rating_distribution = df_cleaned['rating'].value_counts()
plt.figure(figsize=(10, 6))
rating_distribution.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Content Ratings')
plt.show()

# Question 4: How does the runtime vary across different content types (Movies vs. TV Shows)?
# Filter for movies
df_movies = df_cleaned[df_cleaned['type'] == 'Movie']

# Plot movie duration distribution
plt.figure(figsize=(8, 6))
sns.histplot(df_movies['duration'].str.replace(' min', '').astype(int), bins=30)
plt.title('Movie Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of Movies')
plt.show()

# Question 5: What is the correlation between release year and duration?
df_movies['duration'] = df_movies['duration'].str.replace(' min', '').astype(int)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='release_year', y='duration', data=df_movies)
plt.title('Release Year vs. Movie Duration')
plt.xlabel('Release Year')
plt.ylabel('Duration (minutes)')
plt.show()

# Question 6: How does the number of titles produced change over time?
titles_by_year = df_cleaned.groupby('release_year')['title'].count()
plt.figure(figsize=(10, 6))
titles_by_year.plot(kind='line')
plt.title('Number of Titles Released Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.show()

# Question 7: What is the distribution of directors and cast involvement in content creation?
plt.figure(figsize=(10, 6))
directors = df_cleaned['director'].value_counts().head(10)
directors.plot(kind='bar')
plt.title('Top 10 Directors by Number of Titles')
plt.xlabel('Director')
plt.ylabel('Count')
plt.show()

# 4. Data Visualization: Use matplotlib/seaborn visualizations as included above for each question.
