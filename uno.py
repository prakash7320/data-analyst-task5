import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats



df = pd.read_csv("C:/Users/ELCOT/Desktop/train.csv")

df[['Age','Fare','SibSp','Parch']].hist(bins=30, figsize=(12,8))
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1,3, figsize=(16,5))

sns.countplot(x='Pclass', data=df, ax=axes[0])
sns.countplot(x='Sex', data=df, ax=axes[1])
sns.countplot(x='Embarked', data=df, ax=axes[2])

plt.show()
sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Pclass")
plt.show()
sns.kdeplot(df[df['Survived']==1]['Age'].dropna(), shade=True, label='Survived')
sns.kdeplot(df[df['Survived']==0]['Age'].dropna(), shade=True, label='Died')

plt.title("Age Distribution (Survivors vs Non-Survivors)")
plt.legend()
plt.show()
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title("Fare vs Survival")
plt.show()
corr_df = df.copy()

corr_df['Sex'] = corr_df['Sex'].map({'male':0,'female':1})
corr_df['Embarked'] = corr_df['Embarked'].fillna('S').map({'S':0,'C':1,'Q':2})

cols = ['Survived','Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']

plt.figure(figsize=(8,6))
sns.heatmap(corr_df[cols].corr(), annot=True, cmap='coolwarm')
plt.show()
sns.pairplot(df[['Survived','Age','Fare','Pclass']], hue='Survived')
plt.show()
