from data.WorldBankDataLoader import WorldBankDataLoader
import pprint
import pandas
import os
from sklearn import cluster, impute, decomposition
from matplotlib import pyplot as plt
import numpy as np

# get countries in region - here it's Europe & Central Asia
all_countries = WorldBankDataLoader().all_countries()
europe_and_central_asia_countries = {country['name']: None for country in all_countries if
                                     country['region']['id'] == 'ECS'}
for country_name in europe_and_central_asia_countries:
    country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
    europe_and_central_asia_countries[country_name] = pandas.read_csv(country_data_path)

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='mean')
data_per_country = {}
for country_name in europe_and_central_asia_countries:
    data = europe_and_central_asia_countries[country_name][
        ['Population density (people per sq. km of land area)',
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
         ]].fillna(NULL_PLACEHOLDER)
    data_per_country[country_name] = imputer.fit_transform(data)

# cluster countries with k-means
clusterer = cluster.KMeans(n_clusters=6, random_state=42)
# filter out countries without both of expected features (basically Kosovo) - currently hardcoded
featured_countries = [country for (country, data) in data_per_country.items() if data.shape == (60, 5)]
print(len(featured_countries))
# flatten arrays for K-means
X = [data_per_country[country].flatten() for country in featured_countries]
labels = clusterer.fit_predict(X)
grouped_countries = {label: [] for label in labels}
for i in range(len(featured_countries)):
    grouped_countries[labels[i]].append(featured_countries[i])

# show results of clustering
pprint.pprint(grouped_countries)

# display results of clustering by bringing data into 2 dimensions via PCA
pca = decomposition.PCA(n_components=2)
principal_components = pca.fit_transform(X)
x = np.transpose(principal_components)[0]
y = np.transpose(principal_components)[1]
plt.scatter(x, y, c=labels)
plt.show()

# Areas to reconsider:
# - how to deal with NaNs - currently mean is used, perhaps we should consider median or a different idea altogether
# - how to deal with missing features, such as Kosovo's case here - for now the entire country gets skipped
# - how to prepare data for clustering in general - perhaps something different than simple array flattening is desirable?
# - how to cluster data - currently k-means is used
# - how to visualize data - currently PCA is used
