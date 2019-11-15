import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
import shutil


def visualize(centroids, membership_matrix, labels, title="Plot", _plt=plt, colors=None, c_map=None, save_fig=False, dir="temp", filename="fig", marker="x", **kwargs):
    _plt = plt if _plt is None else _plt
    fig = _plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = 8 * ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    # matplotlib consider (x1,x2) and(y1, y2) but our centroids are (x1,y1) (x2,y2)
    # if pca is not None:
    #     principal_component = pca.fit_transform(list(membership_matrix.values()))
    centroids = np.array(centroids).T
    ax.scatter(centroids[0], centroids[1], centroids[2], marker="o", c="K", s=60, depthshade=False)
    for c_id, cluster in enumerate(membership_matrix):
        for _features in membership_matrix[cluster]:
            ax.scatter(_features[0], _features[1], _features[2], marker=marker, linewidths=4, c=colors[c_id],
                       depthshade=False, **kwargs)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    _plt.title(title)
    if(save_fig):
        if(not os.path.exists(dir)):
            os.mkdir(dir)
        _filename= os.path.join(dir, filename)
        _plt.savefig(_filename, bbox_inches="tight")
        print("{} saved in {}".format(filename, dir))
    #plt.show()
    return _plt, ax

def visualize_2(centroids1, centroids2, membership_matrix_1, membership_matrix_2, labels, title="Plot_2",colors=None, c_map=None, save_fig=False, dir="temp", filename="fig", marker="x", **kwargs):
    _plt = plt
    fig = _plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    #colors = 8 * ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    # matplotlib consider (x1,x2) and(y1, y2) but our centroids are (x1,y1) (x2,y2)
    # if pca is not None:
    #     principal_component = pca.fit_transform(list(membership_matrix.values()))
    centroids_1 = np.array(centroids1).T
    centroids_2 = np.array(centroids2).T
    ax.scatter(centroids_1[0], centroids_1[1], centroids_1[2], marker="o", c="K", s=60, depthshade=False)
    ax.scatter(centroids_2[0], centroids_2[1], centroids_2[2], marker="o", c="green", s=60, depthshade=False)

    colors = 8 * ["C1", "C0", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    for c_id, cluster in enumerate(membership_matrix_2):
        for _features in membership_matrix_2[cluster]:
            ax.scatter(_features[0], _features[1], _features[2], marker="o", s=80, alpha=0.2, facecolors='none',linewidths=8, c=colors[c_id],
                       depthshade=False)
    colors = 8 * ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    for c_id, cluster in enumerate(membership_matrix_1):
        for _features in membership_matrix_1[cluster]:
            ax.scatter(_features[0], _features[1], _features[2], marker=marker, linewidths=4, c=colors[c_id],
                       depthshade=False)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    _plt.legend()
    _plt.title(title)
    if(save_fig):
        if(not os.path.exists(dir)):
            os.mkdir(dir)
        _filename= os.path.join(dir, filename)
        _plt.savefig(_filename, bbox_inches="tight")
        print("{} saved in {}".format(filename, dir))
    #plt.show()
    return _plt, ax


def normalize(data):
    #return data
    return ((data - min(data)) / (max(data) - min(data)))

def normalize_columns(dataset):
    df = pd.DataFrame()
    df = pd.DataFrame(columns=dataset.columns)
    for col in dataset.columns:
        #df[col] = ((dataset[col] - min(dataset[col])) / (max(dataset[col]) - min(dataset[col])))
        df[col] = normalize(dataset[col])
    return df


# def int_generator():
#     i += 1
#     yield
