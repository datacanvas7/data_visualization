# %% [markdown]
# # Data Visualization using Seaborn Library
# ## Copy & Learn these graphs from link
# ### https://seaborn.pydata.org/examples/index.html

# %% [markdown]
# ## 01 - lineplot with multiple facets

# %%
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##load data_set
points = sns.load_dataset("dots")
points.head()

# %%
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##load data_set
points = sns.load_dataset("dots")

#defining color palette
palette= sns.color_palette("rocket_r")

#plot #lineplot
sns.replot(data=points,
           x="time", y="firing_rate",
           hue="line", size_order=["T1, T2"], palette=palette,
           height=5, aspect=.75, facet_kws=dict(sharex=False))


# %%
##rectify it
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
points = sns.load_dataset("dots")

# Defining color palette
p = sns.color_palette("rocket_r")  # Corrected function name

# Plot using relplot (corrected function name and syntax)
sns.relplot(data=points, 
            x="time", y="firing_rate", 
            hue="coherence", size="choice", col="align",
            kind="line", size_order=["T1" , "T2"], palette=p,  # Fixed variable reference
            height=5, aspect=0.75, facet_kws=dict(sharex=False),)  # Corrected parentheses


# %% [markdown]
# ## 02- box plot type

# %%
import seaborn as sns 
sns.set_theme(style="ticks", palette="pastel")

#load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m","g"],
            data=tips)
sns.despine(offset=10, trim=True)


# %% [markdown]
# ## 03 - Violin plot

# %%
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the dataset tips
tips = sns.load_dataset("tips")

# Draw a nested violin plot and split the violins for easier comparison
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True, inner="quart", linewidth=1, palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)

# %% [markdown]
# ## 04- Scatterplot for big dataset

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

#laod the diamonds dataset
diamonds =sns.load_dataset("diamonds")

#Draw a scatter plot while assigning point colors and sizes to different
#variables in dataset
f, ax = plt.subplots(figsize=(6.5,6.5))
sns.despine(f,left=True, bottom=True)
clarity_ranking =["I1","SI2", "SI1","VS2","VS1","VVS2","VVS1","IF" ]
sns.scatterplot(x="carat", y="price",
                hue="clarity", size="depth",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=diamonds, ax=ax)

# %% [markdown]
# ## 05- Boxplot & Strip Plot (Boxplot with more details)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")

#Initialize the figure with a logarithmic x axis
f, ax= plt.subplot(figsize=(7, 6))
ax.set_xscale("log")

#load the dataset planets
planets = sns.load_dataset("planets")

# Boxplot the orbital period with horizontal boxes
sns.boxplot(x="distance", y="method", data=planets,
        whis=[0, 100], width=.6, palette="vlag")

# Display points to show each observation
sns.stripplot(x="distance", y="method", data=planets,
          size=4, color=".3", linewidth=0)

# Improve the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

# %%
##after rectification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x-axis
f, ax = plt.subplots(figsize=(7, 6))  # Corrected plt.subplot() to plt.subplots()
ax.set_xscale("log")

# Load the dataset planets
planets = sns.load_dataset("planets")

# Boxplot the orbital period with horizontal boxes
sns.boxplot(x="distance", y="method", data=planets,
            whis=[0, 100], width=.6, palette="vlag")

# Display points to show each observation
sns.stripplot(x="distance", y="method", data=planets,
              size=4, color=".3", linewidth=0)

# Improve the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")  # Removed ylabel for clarity
sns.despine(trim=True, left=True)

# %% [markdown]
# ## 06- Correlational Plot

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the brain_networks dataset
brain_networks = sns.load_dataset("brain_networks", index_col=0)

# Compute the correlation matrix
corr_matrix = brain_networks.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)

# Improve plot aesthetics
plt.title("Correlation Heatmap of Brain Networks", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# %%
##after rectification
import seaborn as sns
import matplotlib.pyplot as plt

# Load a dataset (Iris dataset as a placeholder)
brain_networks = sns.load_dataset("iris")  # Replace with your dataset

# Drop non-numeric columns
brain_networks_numeric = brain_networks.select_dtypes(include=['number'])

# Compute the correlation matrix
corr_matrix = brain_networks_numeric.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)

# Improve plot aesthetics
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)


# %%
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the brain networks dataset, select subset, and collapse the multi-index
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns
                  .get_level_values("network")
                  .astype(int)
                  .isin(used_networks))
df = df.loc[:, used_columns]

df.columns = df.columns.map("-".join)

# Compute a correlation matrix and convert to long-form
corr_mat = df.corr().stack().reset_index(name="correlation")

# Draw each cell as a scatter point with varying size and color
g = sns.relplot(
    data=corr_mat,
    x="level_0", y="level_1", hue="correlation", size="correlation",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(50, 250), size_norm=(-.2, .8),
)

# Tweak the figure to finalize
g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)

# %% [markdown]
# ## 07- Joint and marginal histograms

# %%
import seaborn as sns
sns.set_theme(style="ticks")

# Load the planets dataset and initialize the figure
planets = sns.load_dataset("planets")
g = sns.JointGrid(data=planets, x="year", y="distance", marginal_ticks=True)

# Set a log scaling on the y axis
g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
    sns.histplot, discrete=(True, False),
    cmap="light:#03012d", pmax=.8, cbar=True, cbar_ax=cax
)
g.plot_marginals(sns.histplot, element="step", color="#03012d")

# %% [markdown]
# ## 08- Boxenplot

# %%
import seaborn as sns
sns.set_theme(style="whitegrid")

diamonds = sns.load_dataset("diamonds")
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

sns.boxenplot(
    diamonds, x="clarity", y="carat",
    color="b", order=clarity_ranking, width_method="linear",
)

# %% [markdown]
# ## 09 - Heatmap

# %%
import pandas as pd
import seaborn as sns
sns.set_theme()

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Draw the full plot
g = sns.clustermap(df.corr(), center=0, cmap="vlag",
                   row_colors=network_colors, col_colors=network_colors,
                   dendrogram_ratio=(.1, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))

g.ax_row_dendrogram.remove()

# %% [markdown]
# ## 10- Violinplot

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Load the example dataset of brain network correlations
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Pull out a specific subset of networks
used_networks = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 16, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# Compute the correlation matrix and average over networks
corr_df = df.corr().groupby(level="network").mean()
corr_df.index = corr_df.index.astype(int)
corr_df = corr_df.sort_index().T

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

# Draw a violinplot with a narrower bandwidth than the default
sns.violinplot(data=corr_df, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")

# Finalize the figure
ax.set(ylim=(-.7, 1.05))
sns.despine(left=True, bottom=True)

# %% [markdown]
# ## 11- Scatterplot Matrix

# %%
import seaborn as sns
sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")

# %% [markdown]
# ## 12 - Scatterplot with continuous hues and sizes

# %%
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the example planets dataset
planets = sns.load_dataset("planets")

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
g = sns.relplot(
    data=planets,
    x="distance", y="orbital_period",
    hue="year", size="mass",
    palette=cmap, sizes=(10, 200),
)
g.set(xscale="log", yscale="log")
g.ax.xaxis.grid(True, "minor", linewidth=.25)
g.ax.yaxis.grid(True, "minor", linewidth=.25)
g.despine(left=True, bottom=True)

# %% [markdown]
# ## 13 - Projection Plot

# %%
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# Generate an example radial datast
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), height=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(sns.scatterplot, "theta", "r")

# %% [markdown]
# ## 14 - Dotplot

# %%
import seaborn as sns
sns.set_theme(style="whitegrid")

# Load the dataset
crashes = sns.load_dataset("car_crashes")

# Make the PairGrid
g = sns.PairGrid(crashes.sort_values("total", ascending=False),
                 x_vars=crashes.columns[:-3], y_vars=["abbrev"],
                 height=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(xlim=(0, 25), xlabel="Crashes", ylabel="")

# Use semantically meaningful titles for the columns
titles = ["Total crashes", "Speeding crashes", "Alcohol crashes",
          "Not distracted crashes", "No previous crashes"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

# %% [markdown]
# # Assignment _ Reiteration of dataset
# ## Adapting code & Using Penguins dataset

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Penguins dataset
penguins = sns.load_dataset("penguins")

# Drop missing values to avoid errors
penguins = penguins.dropna()

# Create a Boxen plot
plt.figure(figsize=(8, 6))
sns.boxenplot(data=penguins, x="species", y="body_mass_g", palette="coolwarm")

# Improve plot aesthetics
plt.title("Boxen Plot of Penguin Body Mass by Species", fontsize=14)
plt.xlabel("Penguin Species")
plt.ylabel("Body Mass (g)")

# %%



