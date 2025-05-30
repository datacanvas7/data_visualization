# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#canvas (baloon_board)
sns.set(style="whitegrid")

ship = sns.load_dataset("titanic")

sns.boxplot(x="class", 
                y="fare",
                data="ship")

# %%
##to rectify this error
#dont use data = in "" commas,like  data=ship

#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#canvas (baloon_board)
sns.set(style="whitegrid")

ship = sns.load_dataset("titanic")

sns.boxplot(x="class", 
                y="fare",
                data=ship)

# %%
##load new data set
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt

#canvas (baloon_board)
sns.set(style="whitegrid")

restaurants = sns.load_dataset("tips")

sns.boxplot(x="day", 
                y="tip",
                data=restaurants, saturation=0.1)

# %%
##import numpy 
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy 

#canvas (baloon_board)
sns.set(style="whitegrid")

restaurants = sns.load_dataset("tips")

sns.boxplot(x="day", 
                y="tip",
                data=restaurants, estimator=mean, saturation=0.1)


# %%
## mean or median will not work as BOX plot is already contains quartile

# %%
##import pandas as pd
##import numpy 
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
tip.describe()

# %%
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x=tip["total_bill"])

# %%
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x=tip["total_bill"])
sns.boxplot(y=tip["size"])

# %%
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x="tip", y="day", data=tip)

# %%
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x="tip", y="day",hue="smoker", data=tip)

# %%
##using palette and dodge function
#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x="tip", y="day",hue="smoker", data=tip, palette="Set2", dodge=True)

# %%
##using color hexcodes

#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x="tip", y="day",hue="smoker", data=tip, color="#bc8bf0")

# %%
##for avoid future warnings
##using color hexcodes

#import libraries
import seaborn as sns
import numpy 
import pandas as pd

#canvas (baloon_board)_use to set the background style of plot
sns.set(style="whitegrid")

tip = sns.load_dataset("tips")
sns.boxplot(x="tip", y="day",hue="smoker", data=tip, palette="dark:#bc8bf0")

# %% [markdown]
# ---

# %%
##Customizing plots in python
import seaborn as sns
import pandas as pd
import numpy as np

ship = sns.load_dataset("titanic")
ship.head(4)

# %%
sns.boxplot(x="sex",
           y="fare",
           data=ship)

# %%
sns.boxplot(x="survived",
           y="age",
           data=ship)

# %%
##mean donot exists in boxplot so we will do following steps
sns.boxplot(x="survived",
           y="age", showmeans=True,
           data=ship)

# %%
##for symbols customization for means
sns.boxplot(x="survived",
           y="age", showmeans=True,
            meanprops= {"marker":"*",
                      "markersize":"12",
                      "markeredgecolor": "white"},
           data=ship)

# %%
##labelling the graph
sns.boxplot(x="survived",
           y="age", showmeans=True,
            meanprops= {"marker":"*",
                      "markersize":"12",
                      "markeredgecolor": "white"},
           data=ship)

##show labels
plt.xlabel("How many survived"),
plt.ylabel(" Age(years)"),
plt.title("Box plot of survival")


# %%
##this plt error appears as we forgot to import matplotlib.pyplot library
import matplotlib.pyplot as plt
sns.boxplot(x="survived",
           y="age", showmeans=True,
            meanprops= {"marker":"*",
                      "markersize":"12",
                      "markeredgecolor": "white"},
           data=ship)

##show labels
plt.xlabel("How many survived"),
plt.ylabel(" Age(years)"),
plt.title("Box plot of survival")

# %%
##changing size of these titles and labels
plt.xlabel("How many survived", size=12),
plt.ylabel(" Age(years)", size=12),
plt.title("Box plot of survival", size=14)

##we forgot to import dataset here

# %%
sns.boxplot(x="survived",
           y="age", showmeans=True,
            meanprops= {"marker":"*",
                      "markersize":"12",
                      "markeredgecolor": "white"},
           data=ship)

plt.xlabel("How many survived", size=12),
plt.ylabel(" Age(years)", size=12),
plt.title("Box plot of survival", size=14)

# %% [markdown]
# ## Facet wrap & Facet Grid ?

# %% [markdown]
# ## Facet Grid

# %%
import seaborn as sns

import matplotlib.pyplot as plt

# Load dataset
ship = sns.load_dataset("titanic")

# Create FacetGrid with different categories (e.g., 'sex')
g = sns.FacetGrid(ship, col="sex")  # Creates separate plots for each gender

# Map boxplot to each facet
g.map(sns.boxplot, "survived", "age", order=[0,1], showmeans=True,
      meanprops={"marker": "*", "markersize": 12, "markeredgecolor": "white"})

# Set titles
g.set_axis_labels("How many survived", "Age (years)")
g.fig.suptitle("Box Plot of Survival by Gender", size=8)  # Overall title

# %% [markdown]
# ---
# ## Facet Wrap

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
ship = sns.load_dataset("titanic")

# Create FacetGrid with 'class' as facets and wrap columns
g = sns.FacetGrid(ship, col="class", col_wrap=2)  # Adjust col_wrap to control layout

# Map boxplot to each facet with ordered categories for 'survived'
g.map(sns.boxplot, "survived", "age", order=[0, 1], showmeans=True,
      meanprops={"marker": "*", "markersize": 12, "markeredgecolor": "white"})

# Set labels and title
g.set_axis_labels("Survived", "Age (years)")
g.fig.suptitle("Box Plot of Survival by Passenger Class", size=8)

# %%



