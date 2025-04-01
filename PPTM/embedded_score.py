import os
import time
import pickle
# import umap
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections.abc import Sequence
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from collections.abc import Sequence
from scipy.special import softmax
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, MiniBatchKMeans
# from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA

from . topic_score import Topic_SCORE
from . utils import *

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


class UMAP_Fast:
  def __init__(
    self,
    new_dim: int,
    n_neighbors: int = 10,
    min_dist: float = 0.1,
    fit_prop: float = 0.1,
    points: np.ndarray | None = None,
    npy_path: str | None = None,
    tick: str = "AP",
    pre_pca_reduction: bool = False,
    pca_components: int = 50,
    pca_save_dir: str = None,
    pca_data_save_dir: str = None,
    reduce_dim: int = 0,
    random_seed: int | None = None,
    **kwargs
  ) -> None:
    self.kwargs = kwargs
    self.new_dim = new_dim
    self.n_neighbors = n_neighbors
    self.min_dist = min_dist
    self.tick = tick
    if points is not None:
      self.points = points
    else:
      self.points = np.load(npy_path)[reduce_dim:]
      if pre_pca_reduction:
        print(f"pts before pca have shape:{self.points.shape}")
        self.pca = PCA(n_components=pca_components)
        self.points = self.pca.fit_transform(self.points)[reduce_dim:]
        print(f"finishing pca with acc: {np.cumsum(self.pca.explained_variance_ratio_)[pca_components-1]}")
        plt.plot(np.arange(pca_components),np.cumsum(self.pca.explained_variance_ratio_))
        plt.show()
        # save pca model
        os.makedirs(pca_save_dir, exist_ok=True)
        file_name = f"PCA_n_{pca_components}.pkl"
        save_path = os.path.join(pca_save_dir, file_name)
        with open(save_path, "wb") as file:
          pickle.dump(self.pca,file)
        # save pca transfromed data
        os.makedirs(pca_data_save_dir, exist_ok=True)
        save_file_name = f"word_PCA_n_{pca_components}.npy"
        pca_data_save_path = os.path.join(
          pca_data_save_dir,
          save_file_name
        )
        np.save(pca_data_save_path, self.points)
      
    self.full_size = self.points.shape[0]
    self.fit_prop = fit_prop
    self.fit_size = int(self.full_size * self.fit_prop)
    self.original_dim = self.points.shape[1]
    self.reduce_dim = reduce_dim
    self.pre_pca_reduction = pre_pca_reduction
    self.pca_components = pca_components
    self.umap_model = None
    self.random_seed= random_seed

  def fit_model(
    self,
    load_dir: str = "./pkl",
    load_path: str = "",
    dump_pkl: bool = True,
    dump_dir: str = "./pkl",
    save_dir: str = "./npz",
  ) -> None:
    file_name = f"{self.tick}_umap_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
    if self.pre_pca_reduction:
      file_name = file_name + f"_wPCA_{self.pca_components}"
    if self.reduce_dim > 0:
      file_name = file_name + f"_reduce_dim_{self.reduce_dim}"
    file_name = file_name + f"_seed_{self.random_seed}.pkl"
    if not load_path:
      os.makedirs(load_dir, exist_ok=True)
      load_path = os.path.join(load_dir, file_name)
    if os.path.exists(load_path):
      print("umap model exists, retreving data...")
      dump_pkl = False
      with open(load_path, "rb") as file:
        umap_model = pickle.load(file)
    else:
      # umap
      inds = np.arange(self.full_size)
      if self.random_seed is not None:
        np.random.seed(self.random_seed)
      np.random.shuffle(inds)
      inds = inds[: self.fit_size]
      self.umap_sample_inds = inds # save the sampling pts
      sample_inds_file_name = f"{self.tick}_umap_sample_inds_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
      if self.pre_pca_reduction:
        sample_inds_file_name = sample_inds_file_name + f"_wPCA_{self.pca_components}"
      if self.reduce_dim > 0:
        sample_inds_file_name = sample_inds_file_name + f"_reduce_dim_{self.reduce_dim}"
      sample_inds_file_name = sample_inds_file_name + f"_seed_{self.random_seed}.npy"
      os.makedirs(save_dir, exist_ok=True)
      save_path = os.path.join(save_dir, sample_inds_file_name)
      if self.kwargs.get("save_umap_sample_inds", False):
        np.save(
          save_path,
          self.umap_sample_inds
        )
      fit_points = self.points[inds]
      umap_model = umap.UMAP(
        n_neighbors=self.n_neighbors, min_dist=self.min_dist, n_components=self.new_dim, random_state=self.random_seed
      )
      print(f"pts shape for fitting umap is:{fit_points.shape}")
      umap_model.fit(fit_points)
      print("finishing fitting umap")
    # only save the pickle model.
    if dump_pkl:
      os.makedirs(dump_dir, exist_ok=True)
      dump_path = os.path.join(dump_dir, file_name)
      with open(dump_path, "wb") as file:
        pickle.dump(umap_model, file)
    self.umap_model = umap_model

  def transform_all_points(
    self,
    load_dir: str = "./pkl",
    load_path: str = "",
    dump_pkl: bool = True,
    dump_dir: str = "./pkl",
    save_npy: bool = True,
    save_dir: str = "./npz",
    batch_size: int = 10000,
  ) -> np.ndarray:
    if self.umap_model is None:
      self.fit_model(
        load_dir=load_dir, load_path=load_path, dump_pkl=dump_pkl, dump_dir=dump_dir, save_dir=save_dir
      )

    # NOTE: transform all data thru mini-batches
    # self.new_points = self.umap_model.transform(self.points)
    self.new_points = []
    for i in tqdm(range(0, self.points.shape[0], batch_size), desc="Transforming in mini-batches"):
        batch = self.points[i:i + batch_size]
        transformed_batch = self.umap_model.transform(batch)
        self.new_points.append(transformed_batch)
    self.new_points = np.vstack(self.new_points)
    print("finishing transform word_repr")

    if save_npy:
      os.makedirs(save_dir, exist_ok=True)
      save_file_name = f"{self.tick}_word_umap_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
      if self.pre_pca_reduction:
        save_file_name = save_file_name + f"_wPCA_{self.pca_components}"
      if self.reduce_dim > 0:
        save_file_name = save_file_name + f"_reduce_dim_{self.reduce_dim}"
      save_file_name = save_file_name + f"_seed_{self.random_seed}.npy"
      save_path = os.path.join(
        save_dir,
        save_file_name
      )
      np.save(save_path, self.new_points)

    return self.new_points

  def get_scatter_plot(self, dim1, dim2):
    plt.scatter(self.new_points[:, dim1], self.new_points[:, dim2])


class Embedded_SCORE_C:
  def __init__(
    self,
    num_doc: int,
    doc_lens: Sequence[int] | np.ndarray,
    num_topic: int | None = 3,
    num_hyperword: int | None = 800,
    word_repr: np.ndarray | None = None,
    npy_path: str | None = None,
    dim_repr: int = 10,
    bandwidth: float | None = 0.1,
    tick: str = "AP",
    random_seed: int | None = None,
    clustering_method: str="mini-batch kmeans",
    vertex_hunting_method: str="SPA",
    remove_outlier: bool=True,
    cluster_first: bool=False,
    **kwargs,
  ):
    if word_repr is None:
      word_repr = np.load(npy_path)
    self.num_doc = num_doc
    self.doc_lens = np.array(doc_lens)
    self.num_repr = word_repr.shape[0]
    self.tick = tick
    self.random_seed= random_seed
    # fix seed:
    np.random.seed(random_seed)
    self.umap_kwargs = dict()
    if dim_repr is None or dim_repr >= word_repr.shape[1]:
      # NOTE: dim has been reduced.
      self.dim_repr = word_repr.shape[1]
      self.word_repr = word_repr
    else:
      self.dim_repr = dim_repr
      for key in ["n_neighbors", "min_dist", "fit_prop"]:
        if key in kwargs.keys():
          self.umap_kwargs[key] = kwargs.pop(key)
      umap_get = UMAP_Fast(new_dim=dim_repr, **self.umap_kwargs, points=word_repr, tick=tick, random_seed=self.random_seed)
      self.word_repr = umap_get.transform_all_points() + 0.0
      del umap_get
    self.bandwidth = bandwidth
    self.num_hyperword = num_hyperword
    # self.batch_size = None
    self.num_topic = num_topic
    # self.kmeans = kwargs.get('kmeans', 'kmeans_constrained')
    # self.size_min = kwargs.get('size_min', 15)
    # self.size_max = kwargs.get('size_max', 30)
    self.clustering_method = clustering_method
    self.vertex_hunting_method = vertex_hunting_method
    self.remove_outlier = remove_outlier
    self.cluster_first = cluster_first
    self.kwargs = kwargs

   
  

  def get_hyperword(
    self,
    num_hyperword: int | None = None,
    clustering_method: str | None = None,
    random_seed: int | None = None,
    **kwargs,
  ) -> np.ndarray:
    if num_hyperword is not None:
      self.num_hyperword = num_hyperword
    if clustering_method is not None:
      self.clustering_method = clustering_method
    if random_seed is None:
      random_seed=self.random_seed
    for key in kwargs.keys():
      self.kwargs[key]=kwargs[key]
    print(f"number of hyperwords is {self.num_hyperword}")
    print(f"using {self.clustering_method} to get hyperwords...")
    print(
      "available methods: mini-batch kmeans, kmeans, kmeans-constrained, hierarchical"
    )
    # clustering to get hyperwords
    if self.clustering_method == "mini-batch kmeans":
      batch_size = self.kwargs.get("batch_size", int(self.num_repr / 10))
      num_init = self.kwargs.get("num_init", 1)
      cluster_algo = MiniBatchKMeans(
        n_clusters=self.num_hyperword,
        random_state=random_seed,
        batch_size=batch_size,
        n_init=num_init,
        init='k-means++'
      )
      print(f"begin mini-batch clustering w/ num_init={num_init}")
      self.word_labels = cluster_algo.fit_predict(self.word_repr)
      self.hyperword_repr = cluster_algo.cluster_centers_
      print("finishing mini-batch clustering")
    elif self.clustering_method == "kmeans":
      num_init = self.kwargs.get("num_init", 1)
      # cluster_algo = KMeans(n_clusters=self.num_hyperword, random_state=random_seed, n_init=num_init, init='k-means++')
      cluster_algo = KMeans(n_clusters=self.num_hyperword, random_state=random_seed, n_init=num_init, init='random')
      # init set random, as increasing num_init
      self.word_labels = cluster_algo.fit_predict(self.word_repr)
      self.hyperword_repr = cluster_algo.cluster_centers_
    # elif self.clustering_method == "kmeans-constrained":
    #   cluster_algo = KMeansConstrained(
    #     n_clusters=self.num_hyperword,
    #     size_min=self.kwargs.get("size_min", 2),
    #     size_max=self.kwargs.get("size_max", 5),
    #     random_state=random_seed,
    #   )
      self.word_labels = cluster_algo.fit_predict(self.word_repr)
      self.hyperword_repr = cluster_algo.cluster_centers_
    elif self.clustering_method == "hierarchical":
      Z = linkage(self.word_repr, method="average")
      self.word_labels = fcluster(Z, t=self.num_hyperword, criterion="maxclust") - 1
      unique_labels = np.unique(self.word_labels)
      centroids = np.zeros((unique_labels.shape[0], self.dim_repr))
      for label in unique_labels:
        centroids[label, :] = self.word_repr[self.word_labels == label].mean(axis=0)
      self.hyperword_repr = centroids
    else:
      raise NotImplementedError
    return self.hyperword_repr

  def prepare_matrix(self) -> None:
    self.hyperword_freq = np.zeros((self.num_hyperword, self.num_doc))
    doc_lens_cumsum = np.cumsum(self.doc_lens)
    for doc_k, (doc_len, cumsum) in enumerate(zip(self.doc_lens, doc_lens_cumsum)):
      unique_elements, counts = np.unique(
        self.word_labels[cumsum - doc_len : cumsum], return_counts=True
      )
      self.hyperword_freq[unique_elements,doc_k] = counts
    self.hyperword_freq /= self.hyperword_freq.sum(0)

  def fit_topic_score(
    self,
    num_topic: int | None = None,
    vertex_hunting_method: str | None=None,
    remove_outlier: bool | None=None,
    cluster_first: bool | None=None,
    random_seed: int | None=None,
    **kwargs,
  ) -> tuple:
    if num_topic is not None:
      self.num_topic = num_topic
    if vertex_hunting_method is not None:
      self.vertex_hunting_method=vertex_hunting_method
    if remove_outlier is not None:
      self.remove_outlier=remove_outlier
    if cluster_first is not None:
      self.cluster_first=cluster_first
    if random_seed is None:
      random_seed = self.random_seed
    for key in kwargs.keys():
      self.kwargs[key]=kwargs[key]
    print("In fit_topic_socre:")
    self.model = Topic_SCORE(
      num_doc=self.num_doc,
      num_word=self.num_hyperword,
      word_freq=self.hyperword_freq,
      num_topic=self.num_topic,
      vertex_hunting_method=self.vertex_hunting_method,
      remove_outlier=self.remove_outlier,
      cluster_first=self.cluster_first,
      random_seed=random_seed,
      **self.kwargs,
    )
    # self.hyperword_density, self.W_hat = self.model.fit_all()
    self.model.fit_all()
    self.hyperword_density = self.model.A_hat
    print("finish fitting TS")
    if self.num_topic > 2:
      self.model.plot_vertex_hunting(dim1=0,dim2=1, label1="Hyperwords", show_centroids=True)
    self.hyperword_mixed_member = self.hyperword_density / self.hyperword_density.sum(
      1, keepdims=True
    )
    #return self.hyperword_density, self.hyperword_mixed_member, self.W_hat
  
  def get_word_density(self, bandwidth: float | None = None, down_sample: float | None = None, minibatch: bool = True) -> tuple:
    if bandwidth is not None:
      self.bandwidth = bandwidth
    print(f"bandwidth is {self.bandwidth}")
    if minibatch:
      num_batches = self.kwargs.get("num_batches_cal_density", 5)
      all_batches = np.array_split(np.arange(self.word_repr.shape[0]), num_batches)
      self.word_density = np.zeros((self.word_repr.shape[0],self.num_topic))
      self.word_mixed_member = np.zeros((self.word_repr.shape[0],self.num_topic))
      for batch in tqdm(all_batches, desc="Calculating word densities."):
        
        word_to_hyperword_dist = euc_dist_sq(self.word_repr[batch], self.hyperword_repr)
        gaussian_kernel = np.exp(
          -(word_to_hyperword_dist**2) / self.bandwidth**2
          - np.log(self.bandwidth * np.sqrt(np.pi)) * self.dim_repr
        )
        softmax_kernel = softmax(-(word_to_hyperword_dist**2) / self.bandwidth**2, axis=1)
        self.word_density[batch] = gaussian_kernel @ self.hyperword_density
        pre_mixed_member = softmax_kernel @ self.hyperword_mixed_member
        self.word_mixed_member[batch] = pre_mixed_member / pre_mixed_member.sum(1, keepdims=True)
      self.down_sample_word_index = np.arange(self.word_repr.shape[0])
      return self.word_density, self.word_mixed_member, self.down_sample_word_index
    
    index = np.arange(self.word_repr.shape[0])
    np.random.shuffle(index)
    if down_sample is not None:
      index = index[:int(self.word_repr.shape[0]*down_sample)]
    if bandwidth is not None:
      self.bandwidth = bandwidth
    print(f"bandwidth is {self.bandwidth}")
    word_to_hyperword_dist = euc_dist_sq(self.word_repr[index], self.hyperword_repr)
    gaussian_kernel = np.exp(
      -(word_to_hyperword_dist**2) / self.bandwidth**2
      - np.log(self.bandwidth * np.sqrt(np.pi)) * self.dim_repr
    )
    softmax_kernel = softmax(-(word_to_hyperword_dist**2) / self.bandwidth**2, axis=1)
    self.word_density = gaussian_kernel @ self.hyperword_density
    pre_mixed_member = softmax_kernel @ self.hyperword_mixed_member
    self.word_mixed_member = pre_mixed_member / pre_mixed_member.sum(1, keepdims=True)

    self.down_sample_word_index = index
    return self.word_density, self.word_mixed_member, self.down_sample_word_index

  def fit_all(
    self,
    num_topic: int | None = None,
    num_hyperword: int | None = None,
    bandwidth: float | None = None,
    clustering_method: str | None = None,
    vertex_hunting_method: str | None=None,
    remove_outlier: bool | None=None,
    cluster_first: bool | None=None,
    **kwargs
  ):
    if num_topic is not None:
      self.num_topic = num_topic
    if num_hyperword is not None:
      self.num_hyperword = num_hyperword
    if bandwidth is not None:
      self.bandwidth = bandwidth
    if clustering_method is not None:
      self.clustering_method = clustering_method
    if vertex_hunting_method is not None:
      self.vertex_hunting_method=vertex_hunting_method
    if remove_outlier is not None:
      self.remove_outlier=remove_outlier
    if cluster_first is not None:
      self.cluster_first=cluster_first
    for key in kwargs.keys():
      self.kwargs[key]=kwargs[key]
    self.get_hyperword()
    self.prepare_matrix()
    self.fit_topic_score()
    self.get_word_density()



  def plot_word_vector(self, dim1, dim2):
    pass
    # point_cloud = self.word_mixed_member
    ### removing outliers
    # n_neighbors = 5
    # contamination = 0.1
    # clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    # y_pred = clf.fit_predict(point_cloud)
    # centroids = point_cloud[y_pred == 1]
    # new_index = np.cumsum(y_pred) - 1

    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # axes[0].scatter(
    #   self.hyperword_mixed_member[:, dim1],
    #   self.hyperword_mixed_member[:, dim2],
    # )

    # axes[1].scatter(
    #   self.word_mixed_member[:, dim1],
    #   self.word_mixed_member[:, dim2],
    #   color="lightsteelblue",
    #   label="Outliers",
    #   s=10,
    # )
    # # axes[1].scatter(
    # #   centroids[:, dim1],
    # #   centroids[:, dim2],
    # #   color="cornflowerblue",
    # #   label="Selected",
    # #   s=10,
    # # )
    # # axes[1].scatter(centroids[:, dim1], centroids[:, dim2])

    # plt.show()

  def interactive_word_vector(self):
    pass
    # # only plot the first 3 dimensions
    # side_length = height = 1
    # viz_pts = np.array(
    #   [
    #     [-0.5 * side_length, -height / 3],  # Bottom-left vertex
    #     [0.5 * side_length, -height / 3],  # Bottom-right vertex
    #     [0, 2 * height / 3],  # Top vertex
    #   ]
    # )
    # point_cloud = self.normalized_A_hat @ viz_pts
    # point_cloud_pseudo = self.normalized_A_pseudo @ viz_pts
    # ### removing outliers
    # n_neighbors = 5
    # contamination = 0.1
    # clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    # y_pred = clf.fit_predict(point_cloud)
    # centroids = point_cloud[y_pred == 1]
    # new_index = np.cumsum(y_pred) - 1

    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # axes[0].scatter(point_cloud_pseudo[:, 0], point_cloud_pseudo[:, 1])

    # axes[1].scatter(
    #   point_cloud[:, 0],
    #   point_cloud[:, 1],
    #   color="lightsteelblue",
    #   label="Outliers",
    #   s=2,
    # )
    # axes[1].scatter(
    #   centroids[:, 0], centroids[:, 1], color="cornflowerblue", label="Selected", s=2
    # )
    # axes[1].scatter(centroids[:, 0], centroids[:, 1])

    # plt.show()

  def plot_doc_vector(self, dim1: int = 0, dim2: int = 1):
    pass

  def interactive_doc_vector(self, dim: int = 0, dim2: int = 1):
    pass
  
  def get_top_k_words(
    self, 
    min_occ: int = 5, 
    k: int = 50, 
    word_arr: list | np.ndarray | None = None, 
  ):
    # NOTE: Get top words for each topic.
    lst_elem = []
    lst_indices = []
    min_occ = 5
    for topic in range(self.num_topic):
      arr = word_arr[self.down_sample_word_index][np.argsort(self.word_mixed_member[:, topic])[::-1]]
      unique_elements, first_indices = first_k_reoccurring_elements(arr, k=50, min_occ=min_occ)
      # print(f"Topic {topic+1}: {unique_elements.tolist()}")
      lst_elem.append(unique_elements.tolist())
      lst_indices.append(first_indices)
    return lst_elem, lst_indices

  def get_top_k_docs(
    self, 
    k: int = 50, 
    doc_indices: list | np.ndarray | None = None, 
  ):
    # Return the indices of top k documents for each topic given provided doc_indices.
    if doc_indices is None:
      doc_indices = np.arange(self.num_doc)
    anchor_doc_info = []
    for topic in range(self.num_topic):
      # TODO: Add error for missing model or not fitting W_hat in self.model.
      top_k_ind = doc_indices[np.argsort(self.model.W_hat[topic,:])[::-1][:k]]
      anchor_doc_info.append(top_k_ind)
    return anchor_doc_info
  
  def plot_density_regions(
    self,
    new_bandwidth: int = 0.5,
    seed: int | None = None,
    load_path_umap2: str = "",
    dump_dir_umap2: str = "./pkl", 
    load_path: str = "", # load pivotal variables for visualizing densities
    save_path: str = "", # save pivotal variables for visualizing densities
    region_coords_list1: list | None = None,
    region_coords_list2: list | None = None,
    titles: list | None = None,
    titles_ac: list | None = None,
    save_plot_dir: str = "",
    save_plot_tick: str = "AP",
    epsilon: int = 1e-2,
    batch_size: int | None = None
  ):
    if os.path.exists(load_path):
      with np.load(load_path,) as data:
        x_grid = data['x_grid']
        y_grid = data['y_grid']
        z_grid = data['z_grid']
        w_grid = data['w_grid']
        z_word = data['z_word']
        w_word = data['w_word']
        new_word_repr = data['new_word_repr']
    else:
      if seed is None:
        seed = self.seed
      new_umap_get = UMAP_Fast(
        new_dim=2,
        points=self.word_repr,
        tick=self.tick,
        random_seed=seed * 2,
        **self.umap_kwargs,
      )
      new_umap_get.fit_model(load_path=load_path_umap2, dump_dir=dump_dir_umap2)
      
      if batch_size is not None:
        # transform all pts
        print(f"Transforming all pts with batch size: {batch_size}...")
        new_word_repr = []
        z_word = []
        w_word = []
        for i in tqdm(range(0, self.word_repr.shape[0], batch_size), desc="Transforming in mini-batches"):
            batch = self.word_repr[i:i + batch_size]
            transformed_batch = new_umap_get.umap_model.transform(batch)
            kernel_batch = euc_dist_sq(transformed_batch, new_hyperword_repr)
            kernel_batch = np.exp( 
              -(kernel_batch**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
            )
            z_word_batch = kernel_batch @ self.hyperword_density
            w_word_batch = z_word_batch / (epsilon + z_word_batch.sum(-1, keepdims=True))
            z_word.append(z_word_batch)
            w_word.append(w_word_batch)
            new_word_repr.append(transformed_batch)
        z_word = np.vstack(z_word)
        w_word = np.vstack(w_word)
        new_word_repr = np.vstack(new_word_repr)
      else:
        # transform all pts
        print("Transforming all pts...")
        new_hyperword_repr = new_umap_get.umap_model.transform(self.hyperword_repr)
        new_word_repr = new_umap_get.umap_model.transform(self.word_repr) # ~30s
        kernel = euc_dist_sq(new_word_repr, new_hyperword_repr) # ~30s
        kernel = np.exp(
          -(kernel**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
        )
        z_word = kernel @ self.hyperword_density
        w_word = z_word / (epsilon + z_word.sum(-1, keepdims=True))
      del new_umap_get
      
      # create plot grids
      print("Creating plot grids...")
      xlow, ylow = np.min(new_hyperword_repr, axis=0)
      xhigh, yhigh = np.max(new_hyperword_repr, axis=0)
      delta = 0.05
      x_seq = np.arange(xlow, xhigh, delta)
      y_seq = np.arange(ylow, yhigh, delta)
      x_grid, y_grid = np.meshgrid(x_seq, y_seq)
      dist_grid = euc_dist_sq(np.dstack([x_grid, y_grid]), new_hyperword_repr)
      gaussian_kernel = np.exp(
        -(dist_grid**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
      )
      del dist_grid
      z_grid = gaussian_kernel @ self.hyperword_density
      w_grid = z_grid / (epsilon + z_grid.sum(-1, keepdims=True))
      del gaussian_kernel

      # save unnorm_density (z_word)/ norm_density (w_word)
      if save_path:
        np.savez(
          save_path, 
          z_word=z_word, 
          w_word=w_word, 
          new_word_repr=new_word_repr,
          z_grid=z_grid,
          w_grid=w_grid,
          x_grid=x_grid,
          y_grid=y_grid
        )
    
    viridis = plt.get_cmap('viridis')
    n_colors = 256
    viridis_colors = viridis(np.linspace(0, 1, n_colors))
    new_viridis_colors = viridis(np.linspace(0, 1, n_colors))
    for i in range(0, 256):
      # Reduce the saturation of yellow hues by making it more grayish
      new_viridis_colors[i] = viridis_colors[255-i]*np.array([1, 0.6+0.4*i/256, 0.7+0.3*i/256, 0.85+0.15*i/256])  # Slightly desaturate the RGB values
      new_viridis_colors[i] = viridis_colors[255-i]*np.array([1, 0.9+0.1*(255-i)/256, 0.9+0.1*(255-i)/256, 0.9+0.1*(255-i)/256])  
    custom_cmap = LinearSegmentedColormap.from_list("custom_viridis", new_viridis_colors, N=n_colors)
    cmap_ = mcolors.LinearSegmentedColormap.from_list("CustomGreen", ["white", "green"])

    for topic in range(self.num_topic):
      fig, ax = plt.subplots()
      ax.set_aspect('equal')
      w_grid_masked = np.ma.masked_where(w_grid == 0, w_grid)
      contours = ax.contourf(x_grid, y_grid, w_grid_masked[:, :, topic], cmap=cmap_, zorder=1, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
      contour_lines = ax.contour(x_grid, y_grid, w_grid[:, :, topic], colors='black', linewidths=0.1, zorder=2, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
      
      handles = [
        plt.Line2D([0], [0], color=contours.cmap(contours.norm(level)), lw=2)
        for level in contours.levels
      ]
      labels = [f"{level:.2g}" for level in contours.levels]
      
      if region_coords_list1 is not None and len(region_coords_list1[topic]):
        # Add additional points
        ax.scatter(region_coords_list1[topic][:,0], region_coords_list1[topic][:,1],c='gold',s=125,marker='*',edgecolors='black',linewidths=0.2,zorder=3)
        for idx in range(len(region_coords_list1[topic])):
          ax.text(
              region_coords_list1[topic][idx,0] + 1,
              region_coords_list1[topic][idx,1] - 1,
              region_coords_list2[topic][idx],
              fontsize=16,
              c="magenta",
              bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="magenta", alpha=0.8
              ),
              ha='left',
              va='top',
              zorder=4
            )
      if titles is not None:
        ax.set_title(titles[topic], fontsize=14, fontweight='bold')
      plt.tight_layout()
      if save_plot_dir and save_plot_tick:
        os.makedirs(save_plot_dir, exist_ok=True)
        fig.savefig(
          os.path.join(save_plot_dir, save_plot_tick+f"_contour_normalized_{titles_ac[topic]}.pdf"), format="pdf", bbox_inches="tight"
        )
      
      self.z_word = z_word
      self.w_word = w_word
      self.new_word_repr = new_word_repr
