import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
import torch
import torch.nn.functional as F


get_nll = torch.nn.NLLLoss()


def euclidean_dist(x, y, normalize_by_dim=False):
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  dst = torch.pow(x - y, 2).sum(2)

  if normalize_by_dim:
    dst = dst / float(d)

  return dst


def compute_sinkhorn(m, r=None, c=None, regularization=100., iterations=40):
  '''
  pairwise_distances: (batch, batch')
  r: (batch, dims) distribution (histogram)
  c: (batch', dims) distribution (histogram)
  '''
  # If no distributions are given, consider two uniform histograms
  if r is None:
    r = torch.ones(m.size()[0]).to(m.device) / m.size()[0]
  if c is None:
    c = torch.ones(m.size()[1]).to(m.device) / m.size()[1]

  # Initialize dual variable v (u is implicitly defined in the loop)
  v = torch.ones(m.size()[1]).to(m.device)

  # Exponentiate the pairwise distance matrix
  K = torch.exp(-regularization * m)

  # Main loop
  for i in xrange(iterations):
    # Kdiag(v)_ij = sum_k K_ik diag(v)_kj = K_ij v_j
    # Pij = u_i K_ij v_j
    # sum_j Pij = u_i sum_j K_ij v_j = u_i (Kv)_i = r_i
    # -> u_i = r_i / (Kv)_i
    # K * v[None, :]

    # Match r marginals
    u = r / torch.matmul(K, v)

    # Match c marginals
    v = c / torch.matmul(u, K)

    # print 'P', P
    # print '~r', P.sum(1)
    # print '~c', P.sum(0)
    # print 'u', u
    # print 'v', v
  # Compute optimal plan, cost, return everything
  P = u[:, None] * K * v[None, :]  # transport plan
  dst = (P * m).sum()

  return dst, P, u, v


def log_sum_exp(u, dim):
  # Reduce log sum exp along axis
  u_max, __ = u.max(dim=dim, keepdim=True)
  log_sum_exp_u = torch.log(torch.exp(u - u_max).sum(dim)) + u_max.sum(dim)
  return log_sum_exp_u


def naive_log_sum_exp(u, dim):
  return torch.log(torch.sum(torch.exp(u), dim))


def compute_sinkhorn_stable(m, r=None, c=None, log_v=None, regularization=100., iterations=40):
  # If no distributions are given, consider two uniform histograms
  if r is None:
    r = torch.ones(m.size()[0]).to(m.device) / m.size()[0]
  if c is None:
    c = torch.ones(m.size()[1]).to(m.device) / m.size()[1]
  log_r = torch.log(r)
  log_c = torch.log(c)

  # Initialize dual variable v (u is implicitly defined in the loop)
  if log_v is None:
    log_v = torch.zeros(m.size()[1]).to(m.device)  # ==torch.log(torch.ones(m.size()[1]))

  # Exponentiate the pairwise distance matrix
  log_K = -regularization * m

  # Main loop
  for i in xrange(iterations):
    # Match r marginals
    log_u = log_r - log_sum_exp(log_K + log_v[None, :], dim=1)

    # Match c marginals
    log_v = log_c - log_sum_exp(log_u[:, None] + log_K, dim=0)

  # Compute optimal plan, cost, return everything
  log_P = log_u[:, None] + log_K + log_v[None, :]
  P = torch.exp(log_P)  # transport plan
  dst = (P * m).sum()

  return dst, P, log_P, log_u, log_v


def get_pairwise_distances(m, n):
  assert m.size()[1] == n.size()[1]
  assert len(m.size()) == 2 and len(n.size()) == 2
  distance_matrix = ((m[:, :, None] - n.t()[None, :, :]) ** 2).sum(1)
  return distance_matrix


def cluster_wasserstein_flat(X, n_components, regularization=100., iterations=20, stop_gradient=True, weights=None,
                             add_noise=0.001, sinkhorn_iterations=20, sinkhorn_iterations_warmstart=4):
  '''

  :param X: tensor of shape (n_data, n_dim)
  :param n_components: number of centroids
  :param regularization: 1/regularization in sinkhorn
  :param stop_gradient: whether to cut gradients, if so, centroids are considered to be a
      fixed weighted average of the data. That is the weights (optimal transport plan) are considered not to depend
      on the data.
  :return:
  centroids: tensor of shape (n_components, n_dim)
  P: optimal transport plan
  '''

  assert len(X.size()) == 2, 'Please flatten input to cluster_wasserstein'
  centroids = 0.01 * torch.randn((n_components, X.size()[1])).to(X.device)  # should be fine in most cases
  log_v = None
  for iteration in xrange(iterations):

    distances = get_pairwise_distances(X, centroids)
    # Expectation - Compute Sinkhorn distance
    dst, P, log_P, log_u, log_v = compute_sinkhorn_stable(distances,
      regularization=regularization,
      log_v=log_v,  # warm start after first iteration
      c=weights,  # set cluster masses (None means uniform)
      iterations=sinkhorn_iterations if iteration==0 else sinkhorn_iterations_warmstart)
    soft_assignments = P / P.sum(0, keepdim=True)  # P_ij / sum_i P_ij is soft-assignment of cluster j
    # TODO: maybe dividing by constant will simplify graphs?

    if stop_gradient:
      soft_assignments.detach_()  # how bad is that?

    # Minimization
    centroids = torch.matmul(soft_assignments.t(), X)

    if add_noise > 0:
      centroids.add_(add_noise * torch.randn(centroids.size()).to(X.device))

  return centroids, P


def compute_hungarian(m):
  assert m.size()[0] == m.size()[1]
  m_numpy = m.cpu().detach().numpy()
  row, col = hungarian(m_numpy)
  matrix = np.zeros(m.size())
  matrix[row, col] = 1. / float(len(m))
  cost = (matrix * m_numpy).sum()
  return cost, torch.tensor(matrix), col


def cluster_kmeans_flat(X, n_components, iterations=20, kmeansplusplus=False, epsilon=1e-6):
  '''

  :param X: tensor of shape (n_data, n_dim)
  :param n_components: number of centroids
  '''

  assert len(X.size()) == 2, 'Please flatten input to cluster_wasserstein'

  if not kmeansplusplus:  # initialize from points of dataset
    indices = np.random.choice(range(len(X)), size=n_components, replace=False)
    centroids = X[indices].clone()
  else:  # follow k-means++ initialization scheme
    initial_idx = np.random.randint(len(X))
    centroids_list = []
    for i in xrange(n_components):
      if i == 0:
        new_idx = np.random.randint(len(X))
      else:
        # Compute distances from each point to closest center
        D = get_pairwise_distances(X, centroids)
        D, __ = D.min(1)
        p = D.detach().cpu().numpy()
        p = p / p.sum()  # needs to normalize in numpy for numerical errors
        new_idx = np.random.choice(range(len(X)), p=p)
      centroids_list.append(X[new_idx].clone())
      centroids = torch.stack(centroids_list)

  assignment_matrix = torch.zeros((len(X), n_components)).to(X.device)
  rows = range(len(X))
  for iteration in xrange(iterations):
    # Get  pairwise distances
    distances = get_pairwise_distances(X, centroids)

    # Expectation - Assign each point to closest centroid
    assignment = distances.argmin(1)
    assignment_matrix.fill_(0)
    assignment_matrix[rows, assignment] = 1.

    # Maximization - Average assigned points
    weights = assignment_matrix / (epsilon + assignment_matrix.sum(0))
    centroids = torch.matmul(weights.t(), X)

  return centroids


def cluster_kmeans(X, n_components, iterations=20, kmeansplusplus=False, epsilon=1e-6):
  X_flat = X.view((len(X), -1))
  centroids_flat = cluster_kmeans_flat(X, n_components, iterations, kmeansplusplus, epsilon)
  size = list(X.size())
  size[0] = n_components
  centroids = centroids_flat.view(size)
  return centroids


def clustering_loss(embedded_sample, regularization, clustering_type, normalize_by_dim=True,
                    clustering_iterations=20, sinkhorn_iterations=20, sinkhorn_iterations_warmstart=4,
                    sanity_check=False):
  '''
  This function returns results for two settings (simultaneously):
  - Learning to Cluster:
      - cluster support set.
      - p(y=cluster k | x) given either by Sinkhorn or Softmax
      - reveal support set labels (for evaluation)
      - find optimal matching between predicted clusters and support set labels
      -> Score is clustering accuracy on support set
  - Unsupervised Few-Shot Learning: cluster support set
      - cluster support set, get centroids.
      - p(y=cluster k | x) given either by Sinkhorn or Softmax
      - reveal support set labels (for evaluation)
      - permute clusters accordingly
      - now classify query set data using centroids as prototypes.
      -> Score is supervised accuracy of query set.

  :param embedded_sample:
  :param regularization:
  :param supervised_sinkhorn_loss:
  :param raw_input:
  :return:
  '''

  # Embedding Shapes are (example, dims)
  z_support = embedded_sample['support_embeddings']  # support
  z_query = embedded_sample['query_embeddings']  # query

  # Label Shapes are (example, ways)
  support_labels = embedded_sample['support_labels']
  query_labels = embedded_sample['query_labels']
  support_labels_onehot = embedded_sample['support_labels_onehot']
  query_labels_onehot = embedded_sample['query_labels_onehot']

  n_support = z_support.size(0)
  n_query = z_query.size(0)
  n_class = embedded_sample['way']

  # Compute class counts
  support_counts = support_labels_onehot.sum(0).float()
  support_freq = support_counts / n_support

  # Cluster support set into clusters (both for learning to cluster and unsupervised few shot learning)
  assert clustering_type == 'wasserstein'
  if clustering_type == 'wasserstein':
    z_centroid, __ = cluster_wasserstein_flat(z_support, n_class, regularization=regularization,
      iterations=clustering_iterations, weights=support_freq, stop_gradient=True,
      sinkhorn_iterations=sinkhorn_iterations, sinkhorn_iterations_warmstart=sinkhorn_iterations_warmstart)
  elif clustering_type == 'kmeans':
    z_centroid = cluster_kmeans(z_support, n_class, kmeansplusplus=False)
  elif clustering_type == 'kmeansplusplus':
    z_centroid = cluster_kmeans(z_support, n_class, kmeansplusplus=True)
  else:
    raise Exception('Clustering type not implemented {}'.format(clustering_type))

  # Pairwise distance from query set to centroids
  support_dists = euclidean_dist(z_support, z_centroid, normalize_by_dim)
  query_dists = euclidean_dist(z_query, z_centroid, normalize_by_dim)

  # Assign support set points to centroids (using either Sinkhorn or Softmax)
  all_log_p_y_support = {}
  ############ Sinkhorn conditionals ###################
  # Optimal assignment (could have kept previous result probably)
  __, __, log_assignment, __, __ = compute_sinkhorn_stable(
    support_dists, regularization=regularization, iterations=10)
  # Predictions are already the optimal assignment
  all_log_p_y_support['sinkhorn'] = log_assignment

  ############ Softmax conditionals ###################
  # Unpermuted Log Probabilities
  all_log_p_y_support['softmax'] = F.log_softmax(-support_dists * regularization, dim=1)

  ############ Make predictions in Few-shot clustering (Support) and Unsupervised Few shot learning (Query) mode ###################
  all_support_clustering_accuracy = {}
  all_query_clustering_accuracy = {}
  all_support_losses = {}
  all_query_losses = {}
  for conditional_mode, log_p_y_support in all_log_p_y_support.items():
    # Build accuracy permutation matrix (to match support with ground truth)
    # Shapes:
    #     one_hot_prediction [n_support, n_class_predicted]
    #     support_labels_onehot [n_support, n_class_groundtruth]
    __, y_hat_support = log_p_y_support.max(1)
    y_hat_support = y_hat_support.cpu().numpy()  # to numpy, no need to backprop anyways
    one_hot_prediction = torch.zeros((n_support, n_class))
    one_hot_prediction[range(n_support), y_hat_support] = 1.
    # Shape: accuracy_permutation_cost_support [n_support, n_class_predicted, n_class_groundtruth]
    accuracy_permutation_cost_support = -one_hot_prediction.view(n_support, n_class, 1) * support_labels_onehot.view(n_support, 1, n_class)
    accuracy_permutation_cost_support = accuracy_permutation_cost_support.sum(0)  # add all examples

    # Use Hungarian algorithm to find best match. cols_support[pred_idx] = true_idx
    __, __, cols_support = compute_hungarian(accuracy_permutation_cost_support)
    support_permuted_prediction = cols_support[y_hat_support]
    support_clustering_accuracy = (support_permuted_prediction == support_labels.cpu().numpy().flatten()).mean()

    # Now, run standard prototypical networks, but plugging centroids instead of prototypes
    log_p_y_query = F.log_softmax(-query_dists * regularization, dim=1)
    _, y_hat_query = log_p_y_query.max(1)
    y_hat_query_np = y_hat_query.cpu().numpy()

    # Permute predictions
    query_permuted_predictions = cols_support[y_hat_query_np]
    query_clustering_accuracy = (query_permuted_predictions == query_labels.cpu().numpy().flatten()).mean()

    all_support_clustering_accuracy[conditional_mode] = support_clustering_accuracy
    all_query_clustering_accuracy[conditional_mode] = query_clustering_accuracy

  return {
    'ClusteringAcc_softmax': all_support_clustering_accuracy['softmax'],
    'ClusteringAcc_sinkhorn': all_support_clustering_accuracy['sinkhorn'],
    'UnsupervisedAcc_softmax': all_query_clustering_accuracy['softmax'],
    'UnsupervisedAcc_sinkhorn': all_query_clustering_accuracy['sinkhorn'],
  }
