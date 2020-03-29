from data.WorldBankDataLoader import WorldBankDataLoader
import pprint
import pandas
import os
from sklearn import cluster, impute, decomposition, svm, preprocessing
from matplotlib import pyplot as plt
import numpy as np

# get countries in region - here it's Europe & Central Asia
all_countries = WorldBankDataLoader().all_countries()
europe_and_central_asia_countries = {country['name']:None for country in all_countries if country['region']['id'] == 'ECS'}
for country_name in europe_and_central_asia_countries:
    country_data_path = os.path.join("demography", "downloaded_countries", country_name + ".csv")
    europe_and_central_asia_countries[country_name] = pandas.read_csv(country_data_path)

# this example clusters by population density & urban population in time
# missing values are replaced by mean values of a particular country's data
NULL_PLACEHOLDER = -1
imputer = impute.SimpleImputer(missing_values=NULL_PLACEHOLDER, strategy='median')
used_indicators = ['Population density (people per sq. km of land area)','Urban population (% of total population)']
data_per_country = {}
for country_name in europe_and_central_asia_countries:
    data = europe_and_central_asia_countries[country_name][
            [used_indicators[0],used_indicators[1]]].fillna(NULL_PLACEHOLDER)
    data_per_country[country_name] = imputer.fit_transform(data)

# cluster countries with k-means
clusterer = cluster.KMeans(n_clusters=8, random_state=42)
# filter out countries without both of expected features (basically Kosovo) - currently hardcoded
featured_countries = [country for (country, data) in data_per_country.items() if data.shape==(60,2)]
# flatten arrays for K-means
X = [preprocessing.scale(data_per_country[country], axis=0).flatten(order='F') for country in featured_countries]
labels = clusterer.fit_predict(X)
grouped_countries = {label:[] for label in labels}
for i in range(len(featured_countries)):
    grouped_countries[labels[i]].append(featured_countries[i])

# show results of clustering
pprint.pprint(grouped_countries)

# display results of clustering by bringing data into 2 dimensions via PCA
pca = decomposition.PCA(n_components=2)
principal_components=pca.fit_transform(X)
x = np.transpose(principal_components)[0]
y = np.transpose(principal_components)[1]
plt.scatter(x, y, c=labels)

coeff = np.transpose(pca.components_[0:2, :])


coeff_labels = europe_and_central_asia_countries['Poland']['date'].to_numpy()

n = coeff.shape[0]
for i in range(n):
        plt.arrow(0, 0, coeff[i,0]*40, coeff[i,1]*40,color ='r' if (i<60) else 'g',alpha = 0.5)
        plt.text(coeff[i,0]*40, coeff[i,1]*40, coeff_labels[i % len(coeff_labels)], color = 'b', ha = 'center', va = 'center', size=5)

plot_labels = featured_countries
for i, text in enumerate(plot_labels):
    plt.annotate(text, (x[i], y[i]), ha="center", size=6)

plt.show()

# Areas to reconsider:
# - how to deal with NaNs - currently mean is used, perhaps we should consider median or a different idea altogether
# - how to deal with missing features, such as Kosovo's case here - for now the entire country gets skipped
# - how to prepare data for clustering in general - perhaps something different than simple array flattening is desirable?
# - how to cluster data - currently k-means is used
# - how to visualize data - currently PCA is used