import numpy as np
from scipy.linalg import det
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections.abc import Sequence


def euc_dist_sq(word_repr1, word_repr2):
  return (
    (word_repr1**2).sum(-1, keepdims=True)
    + (word_repr2**2).sum(1)
    - 2 * word_repr1 @ word_repr2.T
  )

# def plot_pseudo_words(word_repr_reduced, num_clusters, random_state, **kwargs):
#   if kwargs.get('kmeans', 'kmeans_constrain') == 'kmeans':
#     kmeans_constrain = KMeans(n_clusters=num_clusters, random_state=random_state)
#   else:
#     kmeans_constrain = KMeansConstrained(
#         n_clusters=num_clusters,
#         size_min=kwargs.get('size_min', 2),
#         size_max=kwargs.get('size_max', 5),
#         random_state=random_state
#     )
#   labels = kmeans_constrain.fit_predict(word_repr_reduced)
#   centroids = kmeans_constrain.cluster_centers_

#   distance_matrix = cdist(word_repr_reduced, centroids, metric='euclidean')

#   plt.figure(figsize=(10, 8))

#   colors = plt.cm.get_cmap('viridis', num_clusters)

#   for cluster_idx in range(num_clusters):
#       cluster_points = word_repr_reduced[labels == cluster_idx]
#       plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
#                   color=colors(cluster_idx))

#   plt.scatter(centroids[:, 0], centroids[:, 1],
#               color='red', marker='x', s=100)

#   plt.plot()
#   return labels, centroids, distance_matrix


def PCA_transform(M, n_components):
  # M as shape (N x p)
  # scikit.learn expect (N x p)
  pca = PCA(n_components=n_components)
  return pca.fit_transform(M)


def NFINDR(data, q: int, transform: np.ndarray | None=None, maxit: int | None=None) -> tuple:
  """
  N-FINDR endmembers induction algorithm.

  Parameters:
    data: `numpy array`
      Column data matrix [nvariables x nsamples].

    q: `int`
      Number of endmembers to be induced.

    transform: `numpy array [default None]`
      The transformed 'data' matrix by MNF (N x components). In this
      case the number of components must == q-1. If None, the built-in
      call to PCA is used to transform the data.

    maxit: `int [default None]`
      Maximum number of iterations. Default = 3*q.


  Returns: `tuple: numpy array, numpy array, int`
    * Set of induced endmembers (N x p)
    * Set of transformed induced endmembers (N x p)
    * Array of indices into the array data corresponding to the
      induced endmembers
    * The number of iterations.

  References:
    Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
    end-member determination in hyperspectral data", presented at the Imaging
    Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.
  """
  # data size
  nsamples, nvariables = data.shape

  if maxit is None:
    maxit = 3 * q

  if transform is None:
    # transform as shape (N x p)
    transform = data
    transform = PCA_transform(data, q - 1)
  else:
    transform = transform

  # Initialization
  # TestMatrix is a square matrix, the first row is set to 1
  TestMatrix = np.zeros((q, q), dtype=np.float32, order="F")
  TestMatrix[0, :] = 1
  IDX = np.zeros((q), dtype=np.int64)
  for i in range(q):
    idx = np.random.randint(nsamples)
    TestMatrix[1:q, i] = transform[idx]
    IDX[i] = idx

  actualVolume = 0
  it = 0
  v1 = -1.0
  v2 = actualVolume

  while it <= maxit and v2 > v1:
    for k in range(q):
      for i in range(nsamples):
        TestMatrix[1:q, k] = transform[i]
        volume = np.abs(det(TestMatrix))
        if volume > actualVolume:
          actualVolume = volume
          IDX[k] = i
      TestMatrix[1:q, k] = transform[IDX[k]]
    it = it + 1
    v1 = v2
    v2 = actualVolume

  E = np.zeros((len(IDX), nvariables), dtype=np.float32)
  Et = np.zeros((len(IDX), q - 1), dtype=np.float32)
  for j in range(len(IDX)):
    E[j] = data[IDX[j]]
    Et[j] = transform[IDX[j]]

  return E, Et, IDX, it


def SPA(R, K):
  n = R.shape[0]

  Y = np.hstack((np.ones((n, 1)), R))
  indexSet = []

  while len(indexSet) < K:
    l2Norms = np.linalg.norm(Y, axis=1)
    index = np.argmax(l2Norms)
    if index not in indexSet:
      indexSet.append(index)
      u = Y[index, :] / np.linalg.norm(Y[index, :])
      Y = Y - np.outer(Y @ u, u)

  return R[indexSet, :], indexSet  # Adjust for 1-based indexing

from scipy.spatial.distance import cdist
def SVS(R, K, quantile=0.9, percentage=0.05):
  # first use knn with 0.05% maximal 
  n = R.shape[0]
  euc_dist = cdist(R, R, metric='euclidean')
  dist_thres = percentage * np.quantile(euc_dist.flatten(), quantile)
  R_knn = np.zeros_like(R)
  for i in range(R.shape[0]):
      neighbors = np.where(euc_dist[i] <= dist_thres)[0]
      R_knn[i] = R[neighbors].mean(axis=0) 

  return SPA(R_knn, K)
  

def get_nearest_point(point_cloud, point):
  return np.argmin(np.linalg.norm(point_cloud - point, axis=1))


# def compute_uci_mixed_membership_documents(doc_topic_probs, topic_probs=None):
#     """
#     Compute UCI score for documents considering mixed membership.

#     :param doc_topic_probs: A matrix where each row represents a document and columns represent the probability
#                             of that document belonging to each topic. Shape: (num_docs, num_topics).
#     :param topic_probs: A list of probabilities of each topic in the corpus. Shape: (num_topics,).

#     :return: UCI score for the set of documents.
#     """
#     num_docs = doc_topic_probs.shape[0]
#     num_topics = doc_topic_probs.shape[1]

#     if topic_probs is None:
#       topic_probs = np.ones(num_topics) / num_topics

#     uci_score = 0
#     pair_count = 0

#     # Iterate over each pair of documents (i, j)
#     for i in range(num_docs):
#         for j in range(i + 1, num_docs):
#             # Marginal probabilities P(d_i) and P(d_j)
#             p_di = np.sum(doc_topic_probs[i, :] * topic_probs)
#             p_dj = np.sum(doc_topic_probs[j, :] * topic_probs)

#             # Joint probability P(d_i, d_j)
#             p_di_dj = np.sum(doc_topic_probs[i, :] * doc_topic_probs[j, :] * topic_probs)

#             # Compute PMI if joint probability > 0
#             if p_di_dj > 0:
#                 uci_score += np.log((p_di_dj + 1e-12) / (p_di * p_dj))
#                 pair_count += 1

#     # Normalize the UCI score by the number of pairs
#     if pair_count > 0:
#         uci_score /= pair_count

#     return uci_score

# def compute_umass_documents(doc_topic_probs, epsilon=1e-12):
#     """
#     Compute UMass coherence score for a topic using a document-topic matrix.

#     :param doc_topic_probs: A matrix where each row represents a document and each column represents
#                             the probability of the document belonging to a particular topic.
#                             Shape: (num_docs, num_topics).
#     :param epsilon: A small constant to avoid division by zero (optional).

#     :return: UMass coherence score for the topic.
#     """
#     num_docs = doc_topic_probs.shape[0]

#     umass_score = 0
#     num_pairs = 0

#     # Iterate over each pair of documents (i, j)
#     for i in range(1, num_docs):
#         for j in range(i):
#             # Marginal probabilities P(d_i) and P(d_j) for the specific topic
#             p_di = np.mean(doc_topic_probs[i, :])  # Average probability across all topics for document i
#             p_dj = np.mean(doc_topic_probs[j, :])  # Average probability across all topics for document j

#             # Joint probability P(d_i, d_j)
#             p_di_dj = np.mean(doc_topic_probs[i, :] * doc_topic_probs[j, :])  # Element-wise product

#             # Compute coherence if p_dj > 0
#             if p_dj > 0:
#                 umass_score += np.log((p_di_dj + epsilon) / p_dj)
#                 num_pairs += 1

#     return umass_score



def first_k_unique_elements(lst: np.ndarray | Sequence, k: int):
  # Convert the list to a NumPy array
  arr = np.array(lst)
  # Preallocate arrays for storing unique elements and their indices
  unique_elements = np.empty(k, dtype=arr.dtype)
  indices = np.empty(k, dtype=int)
  count = 0  # Track the number of unique elements found
  # Iterate through the array
  for idx, elem in enumerate(arr):
    if elem not in unique_elements[:count]:
      unique_elements[count] = elem
      indices[count] = idx
      count += 1
    # Stop once we have found k unique elements
    if count == k:
      break
  if count > k:
    print('fewer than k unique elements were found')
  # Slice the result in case fewer than k unique elements were found
  return unique_elements[:count], indices[:count]


def first_k_reoccurring_elements(lst: np.ndarray | Sequence, k: int, min_occ: int=5):
  # Convert the list to a NumPy array
  arr = np.array(lst)
  # Preallocate arrays for storing unique elements and their indices
  unique_elements = np.empty(len(lst), dtype=arr.dtype)
  reoccurring_elements  = np.empty(k, dtype=arr.dtype)
  occurrence = np.empty(len(lst), dtype=int)
  indices = np.empty(k, dtype=int)
  count_0 = 0  # Track the number of unique elements found
  count_1 = 0
  # Iterate through the array
  for idx, elem in enumerate(arr):
    if elem not in unique_elements[:count_0]:
      unique_elements[count_0] = elem
      occurrence[count_0]=1
      count_0 += 1
    else:
      idx_new = np.argwhere(unique_elements[:count_0]==elem)[0]
      occurrence[idx_new]+=1
      if occurrence[idx_new]== min_occ:
        indices[count_1] = idx
        reoccurring_elements[count_1] = elem
        count_1+=1
    # Stop once we have found k unique elements
    if count_1 == k:
      break
  if count_1 < k:
    print("fewer than k unique elements were found")
  # Slice the result in case fewer than k unique elements were found
  return reoccurring_elements[:count_1], indices[:count_1]


def trace_ind_to_doc(
  indices: np.ndarray | Sequence[int], doc_lens: np.ndarray | Sequence[int]
):
  indices, doc_lens = np.array(indices), np.array(doc_lens)
  doc_lens_cumsum = np.cumsum(doc_lens)
  combined = np.hstack((indices, doc_lens_cumsum))
  sorted_indices = np.argsort(combined)
  index_map = np.zeros_like(sorted_indices)
  index_map[sorted_indices] = np.arange(len(combined))
  doc_indices = index_map[: len(indices)] - np.arange(len(indices))
  return doc_indices


def top_k_docs_info_per_topic(es_model, W_hat, doc_indices, k):
  anchor_doc_info = []
  for topic in range(es_model.num_topic):
    top_k_ind = doc_indices[np.argsort(W_hat[topic,:])[::-1][:k]]
    anchor_doc_info.append(top_k_ind)
  return anchor_doc_info
