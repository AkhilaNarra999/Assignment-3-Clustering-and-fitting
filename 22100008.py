# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 00:49:38 2024

@author: 91905
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Add import statement for StandardScaler
from sklearn.preprocessing import StandardScaler


def plot_correlation_matrix(
        corr_matrix,
        title='Correlation Matrix',
        cmap='magma',
        tick_color='black'):
    """
    Plot the correlation matrix with a heatmap.

    Parameters:
    corr_matrix (pandas.DataFrame): Correlation matrix to plot.
    title (str): Title of the plot.
    cmap (str): Colormap to use for plotting.
    tick_color (str): Color of the ticks on the axes.
"""

    sns.set(style="white")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={
            "shrink": .82})
    plt.xticks(rotation=45, ha='right', color=tick_color)
    plt.yticks(rotation=0, color=tick_color)
    plt.title(title)
    plt.show()


def elbow_method(df_scaled, k_range=range(1, 10)):
    """
    Perform the elbow method to find the optimal number of clusters.

    Parameters:
    df_scaled (pd.DataFrame): Scaled DataFrame.
    k_range (range): Range of k values for the elbow method.

    Returns:
    int: Optimal number of clusters.
    """
    inertia = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Find the optimal k using the elbow method
    optimal_k = 4  # Update this based on your elbow method analysis

    # Plot the elbow plot with a line and markers
    plt.figure(figsize=(10, 6))
    plt.plot(
        k_range,
        inertia,
        marker='o',
        linestyle='-',
        color='b',
        markerfacecolor='r',
        markersize=8)
    plt.scatter(optimal_k,
                inertia[optimal_k - 1],
                c='red',
                marker='x',
                s=100,
                label='Elbow Point')
    plt.xticks(color='Black')
    plt.yticks(color='Black')
    plt.xlabel('Number of Clusters (k)', color='black')
    plt.ylabel('Inertia', color='black')
    plt.title('Elbow Plot to Find Optimal k', color='black')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return optimal_k


def poly_3f(x, a, b, c, d):
    """
    Third-degree polynomial function.

    Parameters:
    x (numpy.ndarray): Input values.
    a, b, c, d (float): Coefficients.

    Returns:
    numpy.ndarray: Result of the polynomial function.
    """
    return a * x**3 + b * x**2 + c * x + d


def poly_2f(x, a, b, c):
    """
    Second-degree polynomial function.

    Parameters:
    x (numpy.ndarray): Input values.
    a, b, c (float): Coefficients.

    Returns:
    numpy.ndarray: Result of the polynomial function.
    """
    return a * x**2 + b * x + c

# Function for curve fitting


def func(x, a, b, c):
    return a * x**2 + b * x + c


def plot_curve_fit(
        xdata,
        ydata,
        yfit,
        sigmas=None,
        model_name='Estimated',
        color='red'):
    """
    Plot the original data with the line of best fit.

    Parameters:
    xdata (numpy.ndarray): Input values.
    ydata (numpy.ndarray): Actual output values.
    yfit (numpy.ndarray): Estimated output values.
    sigmas (numpy.ndarray): Confidence intervals for the fitted curve.
    model_name (str): Name of the model for labeling.
    color (str): Color for plotting.
    """
    plt.scatter(xdata, ydata, label='Actual CO2', color='blue')
    plt.scatter(xdata, yfit, label=f"{model_name} CO2", color=color)
    if sigmas is not None and all(sigmas > 1):
        ylower, yupper = errors.err_ranges(xdata, poly_2f, popt, sigmas)
        plt.fill_between(
            xdata,
            ylower,
            yupper,
            alpha=0.2,
            label='Confidence Range')
    plt.title("Actual and Fitted Data", color='black')
    plt.xlabel('Population', color='black')
    plt.ylabel('CO2 Emission per Capita', color='black')
    plt.grid(color='black', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


# Load data and perform initial processing
pop_df = pd.read_csv('D:\\Aki\\Population.csv')
pop_df = pop_df.melt(
    id_vars='date',
    var_name='country',
    value_name='population')
co2_df = pd.read_csv('D:\\Aki\\CO2 emissions (metric tons per capita).csv')
co2_df = co2_df.melt(
    id_vars='date',
    var_name='country',
    value_name='co2_emission_pc')

pop_df = pop_df.loc[co2_df.co2_emission_pc.notna(), :].reset_index(drop=True)
co2_df = co2_df.loc[co2_df.co2_emission_pc.notna(), :].reset_index(drop=True)
assert co2_df.shape == pop_df.shape, "Unequal number of rows"

df = pd.DataFrame({"population": pop_df.population,
                   'co2_emission_pc': co2_df.co2_emission_pc})

# Scale the data before performing clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['population', 'co2_emission_pc']])

# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot the correlation matrix with the new colormap
plot_correlation_matrix(correlation_matrix)

# Perform clustering
optimal_k = elbow_method(df_scaled)
kmeans = KMeans(n_clusters=optimal_k, n_init='auto')
df['cluster'] = kmeans.fit_predict(df_scaled)

# Visualize clusters and cluster centers
plt.figure(figsize=(10, 8))
colors = ['#f3f847', '#0e4bd2', '#d20eb9', '#0ed210']

for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(
        cluster_data['co2_emission_pc'],
        cluster_data['population'],
        label=f'Cluster {i + 1}',
        color=colors[i])

cluster_centers_backscaled = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(cluster_centers_backscaled[:, 1], cluster_centers_backscaled[:, 0],
            marker='o', s=200, color='black', label='Cluster Centers')

plt.xlabel('CO2 emissions per capita', color='black')
plt.ylabel('Population', color='black')
plt.title('Clusters of Countries', color='black')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7, )
plt.xticks(color='black')
plt.yticks(color='black')
plt.show()

# Analyze Cluster 3 with curve fitting
mask_c3 = df['cluster'] == 2
df_c3 = df[mask_c3].copy()

if not df_c3.empty:
    df_c3_grouped = df_c3.groupby(df_c3.index).agg(
        {'population': 'mean', 'co2_emission_pc': 'mean'}).reset_index()
    popt, pcov = curve_fit(
        func, df_c3_grouped['population'], df_c3_grouped['co2_emission_pc'])
    df_c3_grouped['co2_fit'] = func(df_c3_grouped['population'], *popt)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df_c3_grouped['population'],
        df_c3_grouped['co2_emission_pc'],
        label='Actual Data',
        color='blue')
    plt.plot(
        df_c3_grouped['population'],
        df_c3_grouped['co2_fit'],
        label='Fitted Curve',
        color='red',
        linestyle='--')
    plt.title("Curve Fitting for Cluster 3", color='black')
    plt.xlabel('Population', color='black')
    plt.ylabel('CO2 emission per capita', color='black')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.suptitle('Cluster 3 Analysis', color='black')
    plt.show()

# Analyze Cluster 2 with second-order polynomial curve fitting
mask_c2 = df['cluster'] == 1
df_c2 = df[mask_c2].copy()

if not df_c2.empty:
    df_c2_grouped = df_c2.groupby(df_c2.index).agg(
        {'population': 'mean', 'co2_emission_pc': 'mean'}).reset_index()

    # Perform curve fitting for Cluster 2
    xdata_c2 = df_c2_grouped['population']
    ydata_c2 = df_c2_grouped['co2_emission_pc']
    popt_c2, pcov_c2 = curve_fit(poly_2f, xdata_c2, ydata_c2)
    yfit_c2 = poly_2f(xdata_c2, *popt_c2)

    # Calculate confidence intervals for the fitted curve
    sigmas_c2 = np.sqrt(np.diag(pcov_c2))

    # Scatter plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(
        df_c2['population'],
        df_c2['co2_emission_pc'],
        label='Cluster 2 Data',
        color='blue')
    plt.title("Second Cluster Data", color='black')
    plt.xlabel('Population', color='black')
    plt.ylabel('CO2 Emission per Capita', color='black')
    plt.grid(color='gray', linestyle='--', alpha=0.7)
    plt.legend()

    # Box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(x='cluster', y='co2_emission_pc', data=df, palette='viridis')
    plt.title("Box Plot of CO2 Emission per Capita", color='black')
    plt.xlabel('Cluster', color='black')
    plt.ylabel('CO2 Emission per Capita', color='black')
    plt.grid(color='gray', linestyle='--', alpha=0.7)

    # Violin plot
    plt.subplot(1, 3, 3)
    sns.violinplot(x='cluster', y='co2_emission_pc', data=df, palette='magma')
    plt.title("Violin Plot of CO2 Emission per Capita", color='black')
    plt.xlabel('Cluster', color='black')
    plt.ylabel('CO2 Emission per Capita', color='black')
    plt.grid(color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Plot the curve fit for Cluster 2
    plot_curve_fit(
        xdata_c2,
        ydata_c2,
        yfit_c2,
        sigmas_c2,
        model_name='Estimated',
        color='red')
