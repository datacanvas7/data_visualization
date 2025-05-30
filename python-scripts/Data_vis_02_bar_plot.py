# %% [markdown]
# # Designing Barplot

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a bar plot
sns.barplot(x="species", y="sepal_width", data=phool)
plt.title("Flowers__Bar_Plot")

# %%
phool

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
phool = sns.load_dataset("iris")
phool

#draw a bar plot
sns.barplot(x="species", y="petal_length", data=phool)
plt.title("Flowers__Bar_Plot")

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship)
plt.title("Titanic__Bar_Plot")

# %%
##set order in graph
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship,order=["female", "male"])
plt.title("Titanic__Bar_Plot")

# %% [markdown]
# ## For coloring

# %%
##set order in graph
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship,order=["female", "male"], color = "black")
plt.title("Titanic__Bar_Plot")

# %%
## to avoid warning sign
#Setting a gradient palette using color= is deprecated and will be removed in v0.14.0. Set `palette='dark:black'` for the same effect.
##set order in graph
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship,order=["female", "male"], palette="dark:black")
plt.title("Titanic__Bar_Plot")

# %% [markdown]
# ## remove error bars in graph

# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship,order=["female", "male"], palette="dark:black", ci=None)
plt.title("Titanic__Bar_Plot")

# %%
## To avoid in future use
#The `ci` parameter is deprecated. Use `errorbar=None` for the same effect.

#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="sex", y="class", hue="alive", data=ship,order=["female", "male"], palette="pastel", errorbar=None)
plt.title("Titanic__Bar_Plot")

# %%
##search for builtin palettes for seaborn and build multiple graphs as assignment this week

# %% [markdown]
# ## Estimator
# - Only works for any numeral value 
# - It doesnot work for categorical value

# %%
#changing variables x,y,hue values
#import libraries
import seaborn as sns
#from numpy import mean
#or you can call out library
import numpy
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="class", y="fare", hue="sex", data=ship, estimator = mean)
plt.title("Titanic__Bar_Plot")

# %%
##change color saturation
import seaborn as sns
#from numpy import mean
#or you can call out library
import numpy
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="class", y="fare", hue="sex", data=ship, estimator = mean, saturation=0.2)
plt.title("Titanic__Bar_Plot")



# %%
##to make bar plot horizontal - change x and y values just
import seaborn as sns
#from numpy import mean
#or you can call out library
import numpy
import matplotlib.pyplot as plt

#load dataset
ship = sns.load_dataset("titanic")
ship

#draw a bar plot
sns.barplot(x="fare", y="class", hue="sex", data=ship, saturation=1)
plt.title("Titanic__Bar_Plot")

# %%
##import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#read a titanic csv file
#from seaborn library
ship = sns.load_dataset("titanic")

sns.barplot(x="class", y="fare", data=ship,
            linewidth=2.5, facecolor=(1, 1, 1, 0),
            errcolor="0.2", edgecolor="0.2")

# %%
##again to avoid any error
##import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#read a titanic csv file
#from seaborn library
ship = sns.load_dataset("titanic")

sns.barplot(x="class", y="fare", data=ship,
            linewidth=3, facecolor=(0.5, 0.6, 0.9, 0.8),
            err_kws={'color': '0.6'}, edgecolor="0.2")

# %%



