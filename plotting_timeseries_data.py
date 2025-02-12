import plotly.express as px

def plot_timeline(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title)  # Ensure x and y are keyword arguments
    fig.show()

# Example usage:
plot_timeline(df, x="Month", y="#Passengers", title="Air Passengers Timeline")
