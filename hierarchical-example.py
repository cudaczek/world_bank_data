import os

import pandas
from sklearn import impute

from data.WorldBankDataLoader import WorldBankDataLoader
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import numpy as np

from existence_checker import get_region, verify_region_countries

word_bank_data_loader = WorldBankDataLoader()
all_countries = word_bank_data_loader.all_countries()
all_regions = word_bank_data_loader.regions

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='mean')

DEMOGRAPHIC_INDICATORS = [
    'Population density (people per sq. km of land area)',
    'Urban population (% of total population)',
    "Birth rate, crude (per 1,000 people)",
    "Death rate, crude (per 1,000 people)",
    "Population, male (% of total population)",
    "Sex ratio at birth (male births per female births)",
    "Age dependency ratio (% of working-age population)",
    "Age dependency ratio, old (% of working-age population)",
    "Age dependency ratio, young (% of working-age population)",
    "Urban population (% of total population)",
    "Mortality rate, under-5 (per 1,000 live births)",
    "Fertility rate, total (births per woman)",
    "Population density (people per sq. km of land area)"
]

ECONOMIC_INDICATORS = [
    'GNI per capita',
    'Adjusted savings: education expenditure (% of GNI)',
    'GDP (current US$)',
    # 'Central government debt, total (% of GDP)',
    'Exports of goods and services (% of GDP)',
    'Imports of goods and services (% of GDP)',
    'Final consumption expenditure (% of GDP)',
    'Gross capital formation (% of GDP)',
    'Expense (% of GDP)',
    'Inflation, GDP deflator (annual %)'
]


def get_data_per_country(indicators, region_name, group_name, start_year=None, strategy="mean"):
    data_per_country = {}
    region_data = get_region(region_name, group_name, start_year=start_year)
    region_data = verify_region_countries(indicators, region_data)
    for country_name in region_data:
        data = region_data[country_name][
            [*indicators]].fillna(NULL_PLACEHOLDER)
        imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy=strategy)
        data_per_country[country_name] = imputer.fit_transform(data)
    return data_per_country


def plot_dendrogram(group_name, region, data_per_country, featured_countries, strategy="mean", color_threshold=500):
    X = np.array([np.array(data_per_country[country]).flatten() for country in featured_countries])
    metric = "euclidean"
    method = "ward"
    Z = linkage(X, method, metric=metric)
    fig = plt.figure(figsize=(25, 20))
    plt.title('Hierarchical Clustering Dendrogram ' + region, fontsize=20)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.yscale("log")
    ax = fig.add_subplot(1, 1, 1)
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        # leaf_font_size=10,  # font size for the x axis labels
        labels=np.array(featured_countries),
        color_threshold=color_threshold,
        ax=ax
    )
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.tick_params(axis='y', which='major', labelsize=10)
    plt.savefig(
        group_name + "/dendrogram_regions/" + region + "_hierarchical_" + metric + "_" + method + "_" + strategy + ".png")
    plt.show()


def evaluate_demography():
    group_name = "demography"
    for region in all_regions:
        used_indicators = DEMOGRAPHIC_INDICATORS
        data_per_country = get_data_per_country(used_indicators, region, group_name)
        featured_countries = [country for (country, data) in data_per_country.items() if data.shape == (60, 13)]
        plot_dendrogram(group_name, region, data_per_country, featured_countries)


def evaluate_economy():
    group_name = "economy"
    for region in all_regions:
        if region != 'NA':
            used_indicators = ECONOMIC_INDICATORS
            data_per_country = get_data_per_country(used_indicators, region, group_name, start_year=1989)
            for (country, data) in data_per_country.items():
                print(country, data.shape)
            featured_countries = [country for (country, data) in data_per_country.items() if data.shape == (30, 8)]
            plot_dendrogram(group_name, region, data_per_country, featured_countries, color_threshold=100000000000)


if __name__ == "__main__":
    evaluate_demography()
    evaluate_economy()
