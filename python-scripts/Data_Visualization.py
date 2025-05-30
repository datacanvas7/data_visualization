# %%
#barplot
import seaborn as sns
import matplotlib as plt 
sns.set_theme(style="ticks", color_codes=True)

titanic = sns.load_dataset("titanic")
sns.catplot (x="sex", y="survived", hue= "class", kind="bar", data=titanic)

# %%
#countplot
import seaborn as sns
import matplotlib as plt
sns.set_theme(style="ticks", color_codes=True)
titanic=sns.load_dataset("titanic")
p1=sns.countplot(x="sex", data=titanic, hue="class")
p1.set_title("Plot for counting")

# %%
#scatter_plot
import seaborn as sns
import matplotlib as plt
sns.set_theme(style="ticks", color_codes=True)
titanic=sns.load_dataset("titanic")
g=sns.FacetGrid(titanic, row="sex", hue="alone")
g=(g.map(plt.scatter, "age", "fare").add_legend())

# %%
#after _rectification 
#Error_defined: The error occurred because matplotlib was imported incorrectly
#scatter is not directly available in the base matplotlib module; it should be accessed via matplotlib.pyplot as plt.scatter.
#Just use matplotlib.pyplot instead of matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks", color_codes=True)
titanic=sns.load_dataset("titanic")
g=sns.FacetGrid(titanic, row="sex", hue="alone")
g=(g.map(plt.scatter, "age", "fare").add_legend())


# %% [markdown]
# ### adapt and understand the code by print(dataset) to view it 
# ### make different tables/graph please as assignments 
# ----

# %% [markdown]
# ## day_2 Data Visualization 

# %%
#plt.show() is not for Jupyter notebook its only for VS Code
#Steps involved in data visualization
#step_1_import libraries
import seaborn as sns
import matplotlib.pyplot as plt 

# %%
#step_2_set a theme 
sns.set_theme(style="ticks", color_codes=True)

# %%
#step_3_import data set
ship = sns.load_dataset("titanic")
# print(ship)

# %%
# # #step_4_plot basic graph // only one variable (countplot)
p =sns.countplot(x="sex", data=ship)

# %%
# #step_4_plot basic graph// with two variable (countplot)
p =sns.countplot(x="sex", data=ship, hue="class")

# %%
## #step_4_plot basic graph// with two variable (countplot)_with Title
p =sns.countplot(x="sex", data=ship, hue="class")
p.set_title("Titanic_Countplot")

# %%


# %%



