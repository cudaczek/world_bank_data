import numpy as np
from matplotlib import pyplot as plt, colors, cm
from sklearn import decomposition, manifold


def plot_with_pca(X, assigned_cluster_numbers, point_labels, features_count, coeff_labels):
    pca = decomposition.PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    scatter_points(assigned_cluster_numbers, point_labels, principal_components)
    plot_coeffients(coeff_labels, features_count, pca)
    plt.show()

    print_highest_coefficents(coeff_labels, pca)


def print_highest_coefficents(coeff_labels, pca):
    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    print('Attribute, PC1, PC2:')
    coeffs = sorted([((pc1_val, pc2_val), label) for label, pc1_val, pc2_val in zip(coeff_labels, pc1, pc2)],
                    reverse=True)
    for data in coeffs:
        print(str(data[1]).ljust(70) + ':' + str(repr(data[0][0])).rjust(25) + str(repr(data[0][1])).rjust(25))


def scatter_points(assigned_cluster_numbers, point_labels, principal_components):
    x = np.transpose(principal_components)[0]
    y = np.transpose(principal_components)[1]
    plt.scatter(x, y, c=assigned_cluster_numbers)
    for i, text in enumerate(point_labels):
        plt.annotate(text, (x[i], y[i]), ha="center", size=6)
        if text=="Poland":
            plt.scatter(x[i], y[i], s=100, facecolors='none', edgecolors='r')


def plot_coeffients(coeff_labels, features_count, pca, should_add_labels=False):
    coeff = np.transpose(pca.components_[0:features_count, :])
    single_feature_values_count = len(coeff_labels) // features_count
    # arrows are scaled so that they are visible on a full graph
    coeff_scaling_factor = 40
    cmap = plt.cm.get_cmap(name='jet')
    colors_norm = colors.Normalize(vmin=0, vmax=features_count)
    scalar_map = cm.ScalarMappable(norm=colors_norm, cmap=cmap)
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0] * coeff_scaling_factor, coeff[i, 1] * coeff_scaling_factor,
                  color=scalar_map.to_rgba(i // single_feature_values_count), alpha=0.5)
        if should_add_labels:
           plt.text(coeff[i, 0] * coeff_scaling_factor, coeff[i, 1] * coeff_scaling_factor,
                  coeff_labels[i % len(coeff_labels)], color='b', ha='center', va='center', size=5)


def plot_with_tsne(data, assigned_cluster_numbers, point_labels,perplexity=7, learning_rate=100.0, iterations=20000, should_save=False, filename=''):
    tsne = manifold.TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=iterations)
    # scikit-learn recommends reducing dimensions to about 50 beforehand if there's more of them (e.g. with PCA, so let's do it
    X = data.copy()
    if X[0].shape[0] > 50 and len(X) > 50:
        pca = decomposition.PCA(n_components=50)
        pca = pca.fit_transform(X)

    results = tsne.fit_transform(X)
    scatter_points(assigned_cluster_numbers, point_labels, results)
    plt.title("t-SNE; perplexity={0}; learning rate = {1}, number of iterations = {2}".format(str(perplexity), str(learning_rate), str(iterations)))
    if should_save:
        plt.savefig(filename+'.png')
    else:
        plt.show()