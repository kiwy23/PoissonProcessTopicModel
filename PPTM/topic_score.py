import os
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from collections.abc import Sequence
from scipy.linalg import eigh, pinv, solve
from scipy.cluster.vq import kmeans2
from sklearn.neighbors import LocalOutlierFactor
from cvxopt import matrix, sparse, solvers


from . utils import NFINDR, SPA, SVS
from tqdm import tqdm

solvers.options["show_progress"] = False


# need to double check L1 normalization
# need to implement other vertex hunting algorithms
# need to implement triangle plots


class Topic_SCORE:
  def __init__(
    self,
    num_doc: int,
    num_word: int,
    word_count: np.ndarray | None = None,
    word_freq: np.ndarray | None = None,
    num_topic: int = 3,
    vertex_hunting_method: str = "SPA",
    remove_outlier: bool = True,
    cluster_first: bool = False,
    vocab: Sequence[str] | np.ndarray | None = None,
    tick: str = "AP",
    random_seed: int | None = None,
    row_normalization: bool=False,
    calculate_right_matrix: bool=False, # NOTE: If n large, we suggest setting it to FALSE.
    **kwargs,
  ) -> None:
    self.num_doc = num_doc
    self.num_word = num_word
    if word_count is not None:
      self.doc_lens = word_count.sum(0)
      self.word_freq = word_count / self.doc_lens
      self.row_normalization = row_normalization
    elif word_freq is not None:
      self.word_freq = word_freq
      self.row_normalization = False
    else:
      raise ValueError('need to provide word_freq or word_count')
    
    # self.word_repr = word_repr
    self.vocab = vocab
    # self.bandwidth = None
    # self.word_kernel = None
    # self.word_kernel_inv = None
    
    if calculate_right_matrix:
      self.left_matrix = None
      if self.row_normalization:
        # self.row_sum_sqrt = np.sqrt(self.doc_lens.mean()/self.num_doc*word_count @ (1/self.doc_lens**2)).reshape(-1,1)
        self.row_sum_sqrt = np.sqrt(word_count.sum(axis=-1).reshape(-1,1))
      # self.row_sum_sqrt = np.ones((self.word_freq.shape[0],1))+0.
      # print(self.word_freq.shape, self.row_sum_sqrt.shape)
      # print(min(self.row_sum_sqrt))
        self.normalized_word_freq = word_count/self.row_sum_sqrt
        self.doc_kernel = self.normalized_word_freq.T @ self.normalized_word_freq
      else:
        self.doc_kernel = self.word_freq.T @ self.word_freq
      # eigh is faster than eig as it uses symmetry via symmetric QR; note that here the output eigenvalues are ascending, eigenvectors are the columns of right_matrix
      self.doc_eigen, self.right_matrix = eigh(self.doc_kernel)
      # print(min(self.doc_eigen), max(self.doc_eigen))
    else:
      if self.row_normalization:
        self.row_sum_sqrt = np.sqrt(word_count.sum(axis=-1).reshape(-1,1))
        self.normalized_word_freq = word_count/self.row_sum_sqrt
        self.word_kernel = self.normalized_word_freq @ self.normalized_word_freq.T 
      else:
        self.word_kernel = self.word_freq @ self.word_freq.T
        self.word_eigen, self.left_matrix = eigh(self.word_kernel)
        # print(self.word_eigen)
    
    self.num_topic = num_topic
    self.vertex_hunting_method = vertex_hunting_method
    self.remove_outlier = remove_outlier
    self.cluster_first = cluster_first
    self.tick = tick
    self.random_seed = random_seed
    self.point_cloud = None
    self.vertices = None
    self.centroids = None
    self.kwargs = kwargs

  def get_word_simplex(
    self,
    num_topic: int | None = None,
    vertex_hunting_method: str | None = None,
    remove_outlier: bool | None = None,
    cluster_first: bool | None = None,
    random_seed: int | None = None,
    **kwargs,
  ) -> None:
    if num_topic is not None:
      self.num_topic = num_topic
    if vertex_hunting_method is not None:
      self.vertex_hunting_method = vertex_hunting_method
    if remove_outlier is not None:
      self.remove_outlier = remove_outlier
    if cluster_first is not None:
      self.cluster_first = cluster_first
    if random_seed is None:
      random_seed = self.random_seed
    for key in kwargs.keys():
      self.kwargs[key] = kwargs[key]
    
    if self.left_matrix is None:
      # NOTE: get left_matrix from right_matrix 
      print(f"number of topic is set as {self.num_topic}")
      eigenvalue = self.doc_eigen[: -self.num_topic - 1 : -1]
      right_matrix = self.right_matrix[:, : -self.num_topic - 1 : -1]
      # note: this approach gets rid of matrix square root
      if self.row_normalization:
        self.left_matrix = self.normalized_word_freq @ right_matrix / np.sqrt(eigenvalue)
      else:
        self.left_matrix = self.word_freq @ right_matrix / np.sqrt(eigenvalue)
    else:
      print("left matrix already calculated.")
      eigenvalue = self.word_eigen[: -self.num_topic - 1 : -1]
      print(eigenvalue)
      self.left_matrix = self.left_matrix[: , : -self.num_topic - 1 : -1]
    # ensure the left_matrix start with a positive column; right_matrix should be modified accordingly but we will not do it as right_matrix will never be used later
    if self.left_matrix[:, 0].sum()<0:
      self.left_matrix= -self.left_matrix
    # self.left_matrix[:, 0] = np.abs(self.left_matrix[:, 0])
    self.point_cloud = (self.left_matrix / self.left_matrix[:, (0,)])[:, 1:]
    self.centroids = self.point_cloud + 0.0
    if self.remove_outlier:
      n_neighbors = self.kwargs.get("n_neighbors", 5)
      contamination = self.kwargs.get("contamination", 0.1)
      clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
      self.centroids = self.centroids[clf.fit_predict(self.centroids) == 1]
    if self.cluster_first:
      num_cluster = self.kwargs.get("num_cluster", self.num_topic * 20)
      num_iter = self.kwargs.get("num_iter", self.num_topic * 100)
      self.centroids, _ = kmeans2(
        self.centroids, num_cluster, iter=num_iter, minit="points", seed=random_seed
      )
    # print(self.centroids.shape)
    # plt.scatter(self.centroids[:,0], self.centroids[:,1])
    print(f"using {self.vertex_hunting_method} as the vertex hunting algorithm...")
    if self.vertex_hunting_method == "NFINDR":
      self.vertices = NFINDR(self.centroids, self.num_topic)[0]
    elif self.vertex_hunting_method == "SPA":
      self.vertices = SPA(self.centroids, self.num_topic)[0]
    elif self.vertex_hunting_method == "SVS":
      quantile = self.kwargs.get("quantile", 0.9)
      percentage = self.kwargs.get("percentage", 0.05)
      self.vertices = SVS(self.centroids, self.num_topic, quantile, percentage)[0]
    else:
      raise NotImplementedError

  def get_matrix_factor(self) -> tuple:
    # num_topic = self.left_matrix.shape[1]
    one_vertices = np.hstack([np.ones((self.num_topic, 1)), self.vertices])

    if self.row_normalization:
      A_pre = (self.left_matrix @ pinv(one_vertices)) * self.row_sum_sqrt
    else:
      A_pre = self.left_matrix @ pinv(one_vertices)
    # ensure nonnegative A and normalize the columns
    self.A_hat = np.maximum(A_pre, 0) / np.maximum(A_pre, 0).sum(0)
    # num_topic = A_hat.shape[1]
    
    # note: this approach gets rid of matrix square root
    # also note: using global optimization instead of column-wise optimization is around 10 times faster, using sparse matrix is another 10 times faster, the following is 100 times faster than naive coding!
    # Q = sparse(matrix(np.kron(np.eye(self.num_doc), self.A_hat.T @ self.A_hat)))
    # p = -matrix((self.A_hat.T @ self.word_freq).T.reshape(-1))
    # G = -sparse(matrix(np.eye(self.num_topic * self.num_doc)))
    # h = matrix([0.0] * int(self.num_topic * self.num_doc))
    # A = sparse(matrix(np.kron(np.eye(self.num_doc), np.ones(self.num_topic))))
    # b = matrix([1.0] * int(self.num_doc))
    # self.W_hat = (
    #   np.array(solvers.qp(Q, p, G, h, A, b)["x"])
    #   .reshape(self.num_doc, self.num_topic)
    #   .T
    # )
    if self.kwargs.get("calculate_W_hat", False):
      print("start calculating W_hat")
      num_batches = self.kwargs.get("num_batches_cal_W_hat", 5)
      all_batches = np.array_split(np.arange(self.num_doc), num_batches)
      # all_batches = [batch.tolist() for batch in all_batches]
      self.W_hat = np.zeros((self.num_topic, self.num_doc))
      for batch in tqdm(all_batches):
        batch_size = len(batch)
        Q = sparse(matrix(np.kron(np.eye(batch_size), self.A_hat.T @ self.A_hat)))
        p = -matrix((self.A_hat.T @ self.word_freq[:,batch]).T.reshape(-1))
        G = -sparse(matrix(np.eye(self.num_topic * batch_size)))
        h = matrix([0.0] * int(self.num_topic * batch_size))
        A = sparse(matrix(np.kron(np.eye(batch_size), np.ones(self.num_topic))))
        b = matrix([1.0] * int(batch_size))
        self.W_hat[:, batch] = (
          np.array(solvers.qp(Q, p, G, h, A, b)["x"])
          .reshape(batch_size, self.num_topic)
          .T
        )
      
    self.word_mixed_member = self.A_hat / self.A_hat.sum(-1, keepdims=True)
    # return self.A_hat, self.W_hat
  
  def get_W_hat(self, regression_type="constrained", **kwargs):
    print(f"start calculating W_hat using {regression_type} method")
    num_batches = self.kwargs.get("num_batches_cal_W_hat", 5)
    all_batches = np.array_split(np.arange(self.num_doc), num_batches)
    self.W_hat = np.zeros((self.num_topic, self.num_doc))
    if regression_type == "constrained":
      # all_batches = [batch.tolist() for batch in all_batches]
      for batch in tqdm(all_batches):
        batch_size = len(batch)
        Q = sparse(matrix(np.kron(np.eye(batch_size), self.A_hat.T @ self.A_hat)))
        p = -matrix((self.A_hat.T @ self.word_freq[:,batch]).T.reshape(-1))
        G = -sparse(matrix(np.eye(self.num_topic * batch_size)))
        h = matrix([0.0] * int(self.num_topic * batch_size))
        A = sparse(matrix(np.kron(np.eye(batch_size), np.ones(self.num_topic))))
        b = matrix([1.0] * int(batch_size))
        self.W_hat[:, batch] = (
          np.array(solvers.qp(Q, p, G, h, A, b)["x"])
          .reshape(batch_size, self.num_topic)
          .T
        )
    elif regression_type == "ridge":
      lambda_reg = kwargs.get("lambda_reg", 0.1)
      # Precompute the inverse matrix for ridge regression
      inv_matrix = np.linalg.inv(self.A_hat.T @ self.A_hat + lambda_reg * np.eye(self.num_topic))
      
      for batch in tqdm(all_batches):
        batch_size = len(batch)
        # Ridge regression solution
        W_batch = inv_matrix @ (self.A_hat.T @ self.word_freq[:, batch])
        W_batch[W_batch < 0] = 0
        # Normalize each column to sum to one
        sum_W = np.sum(W_batch, axis=0)
        # Handle cases where the sum is zero
        zero_sum_indices = sum_W == 0
        if np.any(zero_sum_indices):
            print('WARNING: some Ws are 0.')
            W_batch[:, zero_sum_indices] = 1.0 / self.num_topic
            sum_W[zero_sum_indices] = 1.0  # To avoid division by zero
        W_batch /= sum_W  # Normalize columns
        self.W_hat[:, batch] = W_batch

  def fit_all(
    self,
    num_topic: int | None = None,
    vertex_hunting_method: str | None = None,
    remove_outlier: bool | None = None,
    cluster_first: bool | None = None,
    **kwargs,
  ):
    if num_topic is not None:
      self.num_topic = num_topic
    if vertex_hunting_method is not None:
      self.vertex_hunting_method = vertex_hunting_method
    if remove_outlier is not None:
      self.remove_outlier = remove_outlier
    if cluster_first is not None:
      self.cluster_first = cluster_first
    for key in kwargs.keys():
      self.kwargs[key] = kwargs[key]
    self.get_word_simplex()
    return self.get_matrix_factor()

  def plot_vertex_hunting(
    self,
    dim1: int = 0,
    dim2: int = 1,
    vocab: Sequence[str] | np.ndarray | None = None,
    check_list: Sequence[str] | np.ndarray | None = None,
    label1: str = "Words",
    show_centroids: bool = False,
    label2: str = "Centroids",
    save_plot: bool = False,
    save_path: str = "",
  ):
    if vocab is not None:
      self.vocab = vocab
      print("the vocabulary is updated...")
    if check_list is not None:
      if self.vocab is None:
        raise ValueError("need to provide vocabulary to locate the check_list words")
      index_array = np.array(
        [np.where(self.vocab == word) for word in check_list]
      ).reshape(-1)
      # highlighted_pts = centroids[new_index[index_array],:]
      highlighted_pts = self.point_cloud[index_array, :]

    plt.scatter(
      self.point_cloud[:, dim1],
      self.point_cloud[:, dim2],
      color="lightsteelblue",
      label=label1,
      s=10,
    )
    if show_centroids:
      plt.scatter(
        self.centroids[:, dim1],
        self.centroids[:, dim2],
        color="cornflowerblue",
        label=label2,
        s=10,
      )
    plt.scatter(
      self.vertices[:, dim1],
      self.vertices[:, dim2],
      color="red",
      label="Vertices",
      s=20,
    )

    for pair in itertools.combinations(np.arange(self.num_topic), 2):
      plt.plot(self.vertices[pair, dim1], self.vertices[pair, dim2], c="red", ls="--")

    if check_list is not None:
      for point, label in zip(highlighted_pts, check_list):
        plt.scatter(point[dim1], point[dim2], c="brown")
        plt.text(point[dim1], point[dim2], label, fontsize=12, ha="right")

    # note: change limit
    # plt.xlim(-0.2,0.5)
    # plt.ylim(-3,3)
    plt.legend()

    if save_plot:
      if not save_path:
        os.makedirs("./fig", exist_ok=True)
        save_path = os.path.join(
          "./fig", f"{self.tick}_TS_vh_{datetime.datetime.now().strftime('%m%d%H%M%S')}"
        )
      plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()

  def interactive_vertex_hunting(
    self, dim1: int = 0, dim2: int = 1, vocab: Sequence[str] | np.ndarray | None = None
  ):
    if vocab is not None:
      self.vocab = vocab
      print("the vocabulary is updated...")
    word_cloud = pd.DataFrame(
      {
        "word": self.vocab,
        "x": self.point_cloud[:, dim1],
        "y": self.point_cloud[:, dim2],
      }
    )
    fig = px.scatter(word_cloud, x="x", y="y", hover_name="word")

    # note: change limits
    # fig.update_layout(
    #     xaxis=dict(range=[-0.2, 0.5]),  # Set x-axis limits
    #     yaxis=dict(range=[-3, 3])   # Set y-axis limits
    # )
    fig.show()

  def get_anchor_words(self, num: int):
    arr = list()
    for topic in range(self.num_topic):
      arr.append(self.vocab[np.argsort(self.word_mixed_member[:, topic])[::-1]][:num])
      print(arr[-1])
    return np.array(arr)

  # def get_umass(self, A_hat):

  #   self.M = 20
  #   self.M2 = 250
  #   normalized_A_hat = row_normalization(A_hat)
  #   self.normalized_A_hat = normalized_A_hat

  #   ### top words for each topic
  #   self.representative_words = []
  #   for k in range(self.num_topic):
  #       representative_idx = np.argsort(-normalized_A_hat[:,k])
  #       self.representative_words.append([self.vocab[ind] for ind in representative_idx])

  #   top_word_ind = np.zeros_like(normalized_A_hat)
  #   for k in range(self.num_topic):
  #     representative_idx = np.argsort(-normalized_A_hat[:,k])[:self.M2]
  #     top_word_ind[representative_idx,k] = 1
  #   self.diversity = np.sum(np.diag(top_word_ind @ top_word_ind.T ) == 1)/self.M2/self.num_topic

  #   self.pmi = []
  #   for k in range(self.num_topic):
  #     representative_idx = np.argsort(-normalized_A_hat[:,k])[:self.M]
  #     word_freq_repr = self.word_freq_org[representative_idx, :]
  #     co_word_freq_repr = word_freq_repr @ word_freq_repr.T  # (self.M, self.M)
  #     self.pmi.append( np.mean(np.log(((co_word_freq_repr+1)/self.num_doc) / word_freq_repr.
  #                                     mean(1)/ word_freq_repr.mean(1).reshape(-1,1))))

  #   self.tc = []
  #   normalized_A_hat = row_normalization(A_hat)
  #   for k in range(self.num_topic):
  #     representative_idx = np.argsort(-normalized_A_hat[:,k])[:self.M]
  #     word_freq_repr = self.word_freq[representative_idx, :]
  #     co_word_freq_repr = word_freq_repr @ word_freq_repr.T  # (self.M, self.M)
  #     self.tc.append( -np.mean(np.log((co_word_freq_repr + 1e-12) / word_freq_repr.sum(1)/ word_freq_repr.sum(1).reshape
  #                                     (-1,1))/np.log(co_word_freq_repr + 1e-12)) )

  #   word_freq_01 = (self.word_freq != 0)
  #   self.umass = []
  #   for k in range(self.num_topic):
  #       representative_idx = np.argsort(-normalized_A_hat[:,k])[:self.M]
  #       word_freq_repr = word_freq_01[representative_idx, :]
  #       co_word_freq_repr = word_freq_repr @ word_freq_repr.T  # (self.M, self.M)
  #       self.umass.append( np.mean(np.log((co_word_freq_repr + 1) / word_freq_repr.sum(1))) )
