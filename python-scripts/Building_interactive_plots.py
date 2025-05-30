# %% [markdown]
# # Data Visualization using Interactive Plots

# %%
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", marginal_y="violin",
                 marginal_x="box", trendline="ols", template="simple_white")
fig.show()

# %% [markdown]
# ## Animated graphs

# %%
import plotly.express as px
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
                 size="pop", color="continent", hover_name="country", facet_col="continent",
                 log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])
fig.show()

# %% [markdown]
# ## Animated Scatter Plot (Car Share Dataset) Using Chatgpt

# %%
import plotly.express as px
import plotly.io as pio

# Set the renderer for Jupyter Notebook
pio.renderers.default = "notebook"

# Load the Car Share dataset
df = px.data.carshare()

# Create the animated scatter plot
fig = px.scatter(df, x="centroid_lon", y="centroid_lat", animation_frame="peak_hour", 
                 size="car_hours", color="car_hours", hover_name="peak_hour",
                 size_max=50, template="plotly_dark")

# Set axis labels
fig.update_layout(title="Animated Car Share Data Over Peak Hours",
                  xaxis_title="Longitude", 
                  yaxis_title="Latitude")

# Show the figure
fig.show()

# %% [markdown]
# ## Interactive Piechart using chatgpt & codanics

# %%
import plotly.express as px
import pandas as pd

# Example European countries dataset with population
data = {
    "country": ["Germany", "France", "United Kingdom", "Italy", "Spain", "Poland", "Romania", "Netherlands", "Belgium", "Greece"],
    "population": [83166711, 67081000, 68497907, 60262770, 46719142, 38386000, 19237691, 17441139, 11589623, 10423054]
}

# Convert it to a DataFrame
df = pd.DataFrame(data)

# Create the Pie chart
fig = px.pie(df, names="country", values="population", title="Population Distribution of European Countries")

# Show the pie chart
fig.show()

# %%



