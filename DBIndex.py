from itertools import combinations
import  numpy as np
from scipy.spatial import distance


class DBIndex:

    def __init__(self, membership_matrix):
        self.centroids = list(membership_matrix)
        self.membership_matrix = membership_matrix

    # def getDBIndex(self):
    #     pass

    def __inter_cluster_dist(self,membership_matrix, n_pairs=2):
        inter_cluster_dist = {}
        for pair in combinations(membership_matrix, n_pairs):
            # print(pair)
            dist = np.linalg.norm(np.array(pair[0]) - np.array(pair[1]))
            inter_cluster_dist[pair] = dist
            print(dist, distance.euclidean(np.array(pair[0]) , np.array(pair[1])))
            print(np.array_equal(distance.euclidean(np.array(pair[0]), np.array(pair[1])),dist))
        return inter_cluster_dist

    def __with_in_cluster_scatter(self,centroid, observations):
        return np.average(np.linalg.norm(np.array(centroid) - np.array(observations), axis=1))

    def __intra_cluster_dist(self,membership_matrix):
        intra_cluster_dist = {}
        for centroid in membership_matrix:
            intra_cluster_dist[centroid] = self.__with_in_cluster_scatter(centroid, membership_matrix[centroid])
        return intra_cluster_dist

    def __calc_R_i_j(self):
        R_value_metric = {}
        # d(i,j)
        inter_cluster_distances = self.__inter_cluster_dist(self.membership_matrix)
        # s(i) and s(j)
        intra_cluster_distances = self.__intra_cluster_dist(self.membership_matrix)
        R_i_j_dict = {}
        for pairs in combinations(self.membership_matrix, 2):
            S_i = intra_cluster_distances[pairs[0]]
            S_j = intra_cluster_distances[pairs[1]]
            d_i_j = inter_cluster_distances[pairs]
            R_i_j_dict[pairs] = (S_i + S_j) / d_i_j
        return R_i_j_dict



    def compute__DBIndex(self):
        r_i_values = {}
        R_i_j_dict = self.__calc_R_i_j()
        for centroid in self.membership_matrix:  # 1
            r_ij_values = []
            for r_ij in R_i_j_dict:  # 1,2, 1,3
                if centroid in r_ij:
                    r_ij_values.append(R_i_j_dict[r_ij])
            r_i_values[centroid] = max(r_ij_values)
        return np.average(list(r_i_values.values()))


    #def compute_DBI_score(self):





