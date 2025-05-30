# %% [markdown]
# # A Guide to Data Visualization

# %% [markdown]
# ## Introduction
# 
# Seaborn is a Python visualization library built on Matplotlib that provides high-level interfaces for creating statistical graphics. It works seamlessly with Pandas DataFrames and is ideal for exploratory data analysis (EDA).
# 
# ## Why Seaborn?
# 
# ✔ Built-in themes for better aesthetics\
# ✔ Simplified syntax for complex plots\
# ✔ Excellent integration with Pandas\
# ✔ Statistical visualization capabilities

# %% [markdown]
# ---

# %% [markdown]
# ## 01- Installing libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 02- Sample Dataset

# %%
import seaborn as sns
tips = sns.load_dataset('tips')
print(tips.head())

# %% [markdown]
# ## 03- Key Visualizations

# %% [markdown]
# ### A. Distribution Plots
# 
# Aim: Visualize the distribution of a numeric variable.

# %% [markdown]
# 1. Histogram

# %%
sns.histplot(data=tips, x='total_bill', bins=20, kde=True)
plt.title('Distribution of Total Bill Amounts')
plt.show()

# %% [markdown]
# - Use: Shows frequency distribution of total_bill.
# - Limitations:
# 
# Bin size selection can distort interpretation.
# 
# Not ideal for small datasets.

# %% [markdown]
# 2. KDE Plot

# %%
sns.kdeplot(data=tips, x='total_bill', shade=True)
plt.title('Density Estimate of Total Bill')
plt.show()

# %% [markdown]
# - Use: Smooth version of a histogram.
# - Limitations:
# 
# Can oversmooth small datasets.
# 
# 

# %% [markdown]
# ---

# %% [markdown]
# ### B. Categorical Plots
# Aim: Compare categories.

# %%
sns.barplot(data=tips, x='day', y='total_bill', ci=None)
plt.title('Average Bill by Day')
plt.show()

# %% [markdown]
# - Use: Compares means across categories.
# - Limitations:
# 
# Hides underlying data distribution.

# %% [markdown]
# 2. Box Plot

# %%
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('Bill Distribution by Day and Gender')
plt.show()

# %% [markdown]
# - Use: Shows median, quartiles, and outliers.
# - Limitations:
# 
# Less intuitive for non-technical audiences.

# %% [markdown]
# ---

# %% [markdown]
# ### C. Relational Plots
# Aim: Explore relationships between variables.

# %% [markdown]
# 1. Scatter Plot

# %%
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
plt.title('Tip vs. Total Bill by Meal Time')
plt.show()

# %% [markdown]
# - Use: Reveals correlations between two numeric variables.
# - Limitations:
# 
# Overplotting with large datasets

# %% [markdown]
# 2. Line Plot
# 

# %%
sns.lineplot(data=tips, x='size', y='total_bill', ci=95)
plt.title('Bill Amount by Group Size')
plt.show()

# %% [markdown]
# - Use: Trends over a continuous variable.
# - Limitations:
# 
# Assumes ordered data.

# %% [markdown]
# ---

# %% [markdown]
# D. Matrix Plots

# %% [markdown]
# Aim: Visualize matrix-like data.

# %% [markdown]
# 1. Heatmap

# %%
sns.heatmap(tips.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
# Calculate correlation for numeric columns only
numeric_tips = tips.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_tips.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numeric Columns Only)')
plt.show()

# %% [markdown]
# - Use: Identifies correlations between numeric variables.
# - Limitations:
# 
# Only works with numeric data.

# %% [markdown]
# ---

# %% [markdown]
# ## 04- Advanced Plots

# %% [markdown]
# 1. Pair Plot

# %%
sns.pairplot(tips, hue='sex', corner=True)
plt.suptitle('Pairwise Relationships')
plt.show()

# %% [markdown]
# - Use: Auto-generates scatter plots for all numeric columns.
# - Limitations:
# 
# Computationally heavy for large datasets.

# %% [markdown]
# 2. Violin Plot

# %%
sns.violinplot(data=tips, x='day', y='tip', split=True, hue='sex')
plt.title('Tip Distribution by Day and Gender')
plt.show()

# %% [markdown]
# - Use: Combines box plot and KDE.
# - Limitations:
# 
# Can be harder to interpret.

# %% [markdown]
# ---

# %% [markdown]
# ## 05- Customization

# %%
# Style and context
sns.set_style('darkgrid')
sns.set_context('talk')

# Color palettes
sns.set_palette('pastel')

# %% [markdown]
# ##  06- Saving Plots

# %%
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# %% [markdown]
# ## 07- Limitations of Seaborn
# - Not Interactive: Unlike Plotly, plots are static.
# 
# - Limited 3D Support: Primarily for 2D visualizations.
# 
# - Steep Learning Curve: Advanced customizations require Matplotlib knowledge.

# %% [markdown]
# ---


