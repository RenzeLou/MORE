from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import torch


def euclidean_dist_rank(x, y):
    """
    The original function come from : https://github.com/Qidian213/Ranked_Person_ReID

    Args:
      x: pytorch Tensor, with shape [m, d]
      y: pytorch Tensor, with shape [n, d]
    Returns:
      dist: pytorch Tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


def model_pred(datasets, pred_vector, opt):
    # datasets:a list or a 3D numpy array
    feature_embeddings = []
    for sample in datasets:
        vector = pred_vector(sample.reshape(1, 3, -1), opt)  # [1,embedding_size]
        vector = vector.detach().cpu().numpy().tolist()
        feature_embeddings.append(vector[0])
    feature_embeddings = np.array(feature_embeddings)

    return feature_embeddings


def K_means(datasets, pred_vector, num_classes, opt):
    '''
    Args:
        datasets: a list ,each element is a [3,max_len] array sample
        pred_vector: a model's function to predict embedding
        num_classes: num of class

    Returns:
        K-means results -- a tuple(label_list, message, cluster_centers, features)
    '''
    feature_embeddings = model_pred(datasets, pred_vector, opt)
    # print("feature ：",feature_embeddings)
    # print("num class:",num_classes)
    kmeans = KMeans(n_clusters=num_classes, n_init=10).fit(feature_embeddings)
    label_list = kmeans.labels_.tolist()
    # print("cluster num:",len(np.unique(label_list)))
    return label_list, create_msg(label_list), kmeans.cluster_centers_, feature_embeddings


def mean_shift(datasets, pred_vector, opt):
    '''
    same as K_means
    '''
    feature_embeddings = model_pred(datasets, pred_vector, opt)
    # band_width = estimate_bandwidth(feature_embeddings,quantile=0.5,n_samples=800)
    meanshift = MeanShift(bandwidth=opt.band).fit(feature_embeddings)
    label_list = meanshift.labels_.tolist()
    return label_list, create_msg(label_list), meanshift.cluster_centers_, feature_embeddings


def create_msg(label_list):
    # print('creating cluster messages...')
    msg = {}
    msg['num_of_nodes'] = len(label_list)
    msg['num_of_clusters'] = len(list(set(label_list)))
    msg['num_of_data_in_clusters'] = {}
    for reltype in label_list:
        try:
            msg['num_of_data_in_clusters'][reltype] += 1
        except:
            msg['num_of_data_in_clusters'][reltype] = 1

    return msg


def create_my_data(num, shape):
    # create data like val_data or test_data(use to debug)
    print("creating debug data...")
    my_data = []
    for i in range(num):
        t = np.random.randint(200, size=shape)
        my_data.append(t)

    my_label = np.random.randint(100, size=(num)).tolist()
    return my_data, my_label
