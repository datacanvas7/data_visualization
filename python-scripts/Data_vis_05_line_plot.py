# %% [markdown]
# # import libraries
# Seaborn automatically install these libraries
# - numpy
# - scipy
# - pandas
# - matplotlib

# %% [markdown]
# ## adding titles

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a line plot
sns.lineplot(x="sepal_length", y="sepal_width", data=phool)
plt.title("Flowers_Plot")

# %% [markdown]
# ## adding limits

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a line plot
sns.lineplot(x="sepal_length", y="sepal_width", data=phool)
plt.title("Flowers_Plot")
plt.xlim(2)
plt.ylim(2)

# %% [markdown]
# ## Set styles
# - darkgrid
# - whitegrid
# - dark
# - white
# - ticks

# %%
set_style(style=None, rc=None)

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a line plot
sns.lineplot(x="sepal_length", y="sepal_width", data=phool)
plt.title("Flowers_Plot")
sns.set_style("dark")

# %% [markdown]
# ## how to rectify this error

# %%
##set style error
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a line plot
sns.lineplot(x="sepal_length", y="sepal_width", data=phool)
plt.title("Flowers_Plot")

#set style
sns.set_style("dark")
#set_style(style=None, rc=None) #sns.set_style("dark") already sets the style of the plot

#Calling set_style() again without importing or defining it results in a NameError.

# %% [markdown]
# ## size of figure

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#change figure
plt.figure(figsize=(8,6))

#draw a line plot
sns.lineplot(x="sepal_length", y="sepal_width", data=phool)
plt.title("Flowers_Plot")

# %%



