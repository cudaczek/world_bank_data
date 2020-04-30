import os
import pprint

import pandas
from sklearn import cluster, impute, preprocessing
import itertools
from data.WorldBankDataLoader import WorldBankDataLoader
from plot_clusters import plot_with_pca, plot_with_tsne
from existence_checker import verify_region_countries


def get_data_for_region_per_country(region_name):
    all_countries = WorldBankDataLoader().all_countries()
    selected_countries = {country['name']: None for country in all_countries if country['region']['id'] == region_name}
    for country_name in selected_countries:
        country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
        selected_countries[country_name] = pandas.read_csv(country_data_path)
    return selected_countries


def replace_nulls(country_data):
    NULL_PLACEHOLDER = -1
    imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='median')
    data_per_country = {}
    for country_name in country_data:
        data = country_data[country_name][[*used_indicators]].fillna(NULL_PLACEHOLDER)
        data_per_country[country_name] = imputer.fit_transform(data)
    return data_per_country


countries_data_to_show = get_data_for_region_per_country('MEA')
dates = next(iter(countries_data_to_show.values()))['date'].to_numpy()  # any country would suffice, we just need dates
used_indicators = list(WorldBankDataLoader().demographic_indicators().values())
countries_data_to_show = verify_region_countries(used_indicators, countries_data_to_show)

data_per_country_without_nulls = replace_nulls(countries_data_to_show)

# cluster countries with k-means
clusterer = cluster.KMeans(n_clusters=8, random_state=42)
# filter out countries without all expected features
featured_countries = [country for (country, data) in data_per_country_without_nulls.items() if
                      data.shape == (len(dates), len(used_indicators))]
# scale feature; flatten arrays for K-means
X = [preprocessing.scale(data_per_country_without_nulls[country], axis=0).flatten(order='F') for country in
      featured_countries]

labels = clusterer.fit_predict(X)
grouped_countries = {label: [] for label in labels}
for i in range(len(featured_countries)):
    grouped_countries[labels[i]].append(featured_countries[i])

# show results of clustering
pprint.pprint(grouped_countries)

feature_names = [indicator + "(" + str(year) + ")" for (indicator, year) in itertools.product(used_indicators, dates)]

# display results of clustering by:
#  bringing data into 2 dimensions via PCA
#plot_with_pca(X, labels, featured_countries, len(used_indicators), feature_names)
# using t-SNE
plot_with_tsne(X, labels, featured_countries)
