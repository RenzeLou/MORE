import numpy as np
import copy
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, \
    v_measure_score
import torch
import torch.nn.functional as F
from utils import tool_box as tl
from config import Option

'''
The original edition come from : https://github.com/CongWeilin/cluster-loss-tensorflow
And I rewrite it with pytorch
'''


class Facility(object):
    def __init__(self):
        super(Facility, self).__init__()

    def compute_nmi_score(self, labels, predictions):
        # caculate nmi score to measure the clustering quality
        # input must be 1D list or array with int element
        return normalized_mutual_info_score(labels, predictions)

    def compute_ami_score(self, labels, predictions):
        # caculate ami score to measure the clustering quality
        # input must be 1D list or array with int element
        return adjusted_mutual_info_score(labels, predictions)

    def compute_ari_score(self, labels, predictions):
        # ari score can go below 0
        # http://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-score
        ari_score = adjusted_rand_score(labels, predictions)
        return max(0.0, ari_score)

    def compute_vmeasure_score(self, labels, predictions):
        # v_measure score can go below 0
        vmeasure_score = v_measure_score(labels, predictions)
        return max(0.0, vmeasure_score)

    def compute_zeroone_score(self, labels, predictions):
        zeroone_score = float(np.sum((labels == predictions).astype(int)) == len(labels))
        return zeroone_score

    def compute_clustering_score(self, labels, predictions, margin_type):
        """Computes the clustering score via sklearn.metrics functions.

        There are various ways to compute the clustering score. Intuitively,
        we want to measure the agreement of two clustering assignments (labels vs
        predictions) ignoring the permutations and output a score from zero to one.
        (where the values close to one indicate significant agreement).
        This code supports following scoring functions:
          nmi: normalized mutual information
          ami: adjusted mutual information
          ari: adjusted random index
          vmeasure: v-measure
          const: indicator checking whether the two clusterings are the same.
        See http://scikit-learn.org/stable/modules/classes.html#clustering-metrics
          for the detailed descriptions.
        Args:
          labels: 1-D Tensor. ground truth cluster assignment.
          predictions: 1-D Tensor. predicted cluster assignment.
          margin_type: Type of structured margin to use. Default is nmi.
        Returns:
          clustering_score: dtypes.float32 scalar.
            The possible valid values are from zero to one.
            Zero means the worst clustering and one means the perfect clustering.
        Raises:
          ValueError: margin_type is not recognized.
        """
        if margin_type == 'nmi':
            score = self.compute_nmi_score(labels.cpu(), predictions.cpu())
        elif margin_type == 'ami':
            score = self.compute_ami_score(labels.cpu(), predictions.cpu())
        elif margin_type == 'ari':
            score = self.compute_ari_score(labels.cpu(), predictions.cpu())
        elif margin_type == 'vmeasure':
            score = self.compute_vmeasure_score(labels.cpu(), predictions.cpu())
        elif margin_type == 'const':
            score = self.compute_zeroone_score(labels.cpu(), predictions.cpu())
        else:
            raise ValueError('Unrecognized margin_type: %s' % margin_type)

        return score

    def compute_gt_cluster_score(self, pairwise_distances, labels):
        """Compute ground truth facility location score.

        Loop over each unique classes and compute average travel distances.

        Args:
          pairwise_distances: 2-D numpy array of pairwise distances.
          labels: 1-D numpy array of ground truth cluster assignment.

        Returns:
          gt_cluster_score: dtypes.float32 score.
        """
        unique_class_ids = torch.unique(labels)
        num_classes = len(unique_class_ids)
        gt_cluster_score = tl.cudafy(torch.from_numpy(np.array([0.0])))[0]

        for i in range(num_classes):
            """Per each cluster, compute the average travel distance."""
            mask = labels == unique_class_ids[i]
            this_cluster_ids = torch.where(mask)[0]
            temp = (tl.gather(pairwise_distances, this_cluster_ids)).T
            pairwise_distances_subset = (tl.gather(temp, this_cluster_ids)).T
            this_cluster_score = -1.0 * torch.min(torch.sum(pairwise_distances_subset, 0))
            gt_cluster_score += this_cluster_score

        return gt_cluster_score

    def get_cluster_assignment(self, pairwise_distances, centroid_ids):
        """Assign data points to the neareset centroids.

        Tensorflow has numerical instability and doesn't always choose
          the data point with theoretically zero distance as it's nearest neighbor.
          Thus, for each centroid in centroid_ids, explicitly assign
          the centroid itself as the nearest centroid.
          This is done through the mask tensor and the constraint_vect tensor.

        Args:
          pairwise_distances: 2-D Tensor of pairwise distances.
          centroid_ids: 1-D Tensor of centroid indices.

        Returns:
          y_fixed: 1-D tensor of cluster assignment.
        """
        # t1=time.time()
        chose_matrix = tl.gather(pairwise_distances, centroid_ids).cpu()
        predictions = torch.argmin(chose_matrix, 0)  # Tensor int [batch,]
        # for i, ids in enumerate(centroid_ids):
        #     predictions[ids] = i
        # t2=time.time()

        # Deal with numerical instability
        oneHot = F.one_hot(centroid_ids, pairwise_distances.shape[0])
        mask = torch.sum(oneHot, 0)
        mask_ans = torch.mul((~mask.bool()).int(), predictions)
        range_tensor = torch.arange(len(centroid_ids))
        range_tensor = range_tensor.repeat(pairwise_distances.shape[0], 1)
        temp = torch.mul(range_tensor.T, oneHot)
        y_fixed = torch.sum(temp, 0) + mask_ans

        # mask = math_ops.reduce_any(array_ops.one_hot(
        #     centroid_ids, batch_size, True, False, axis=-1, dtype=dtypes.bool),
        #     axis=0)
        # constraint_one_hot = math_ops.multiply(
        #     array_ops.one_hot(centroid_ids,
        #                       batch_size,
        #                       array_ops.constant(1, dtype=dtypes.int64),
        #                       array_ops.constant(0, dtype=dtypes.int64),
        #                       axis=0,
        #                       dtype=dtypes.int64),
        #     math_ops.to_int64(math_ops.range(array_ops.shape(centroid_ids)[0])))
        # constraint_vect = math_ops.reduce_sum(
        #     array_ops.transpose(constraint_one_hot), axis=0)
        #
        # y_fixed = array_ops.where(mask, constraint_vect, predictions)
        # return y_fixed

        return y_fixed

    def _find_loss_augmented_facility_idx(self, pairwise_distances, labels, chosen_ids,
                                          candidate_ids, margin_multiplier,
                                          margin_type):
        """Find the next centroid that maximizes the loss augmented inference.

        This function is a subroutine called from compute_augmented_facility_locations

        Args:
          pairwise_distances: 2-D Tensor of pairwise distances.
          labels: 1-D Tensor of ground truth cluster assignment.
          chosen_ids: 1-D Tensor of current centroid indices.
          candidate_ids: 1-D Tensor of candidate indices.
          margin_multiplier: multiplication constant.
          margin_type: Type of structured margin to use. Default is nmi.

        Returns:
          integer index.
        """
        num_candidates = len(candidate_ids)

        pairwise_distances_candidate = tl.gather(pairwise_distances, candidate_ids)

        if len(chosen_ids) == 0:
            temp = torch.min(-pairwise_distances_candidate.reshape(1, -1), 0)[0]
        else:
            pairwise_distances_chosen = tl.gather(pairwise_distances, chosen_ids)
            pairwise_distances_chosen_tile = pairwise_distances_chosen.repeat(1, num_candidates)
            temp = \
                torch.min(-torch.cat((pairwise_distances_chosen_tile, pairwise_distances_candidate.reshape(1, -1)), 0),
                          0)[
                    0]
        # [num_chose, num_candicates * batch]

        # temp:[num_candidate * embedding] 1D Tensor
        candidate_scores = -1.0 * torch.sum(temp.reshape(num_candidates, -1), 1)
        candidate_scores = candidate_scores.cpu()

        argmax_index = np.argmax(candidate_scores.detach().numpy())

        return candidate_ids[argmax_index]

    def compute_augmented_facility_locations(self, pairwise_distances, labels,
                                             all_ids, margin_multiplier,
                                             margin_type):
        """Computes the centroid locations.

          Args:
            pairwise_distances: 2-D Tensor of pairwise distances.
            labels: 1-D Tensor of ground truth cluster assignment.
            all_ids: 1-D numpy array of all data indices.
            margin_multiplier: multiplication constant.
            margin_type: Type of structured margin to use. Default is nmi.

          Returns:
            chosen_ids: 1-D Tensor of chosen centroid indices.
          """
        num_classes = len(torch.unique(labels))
        chosen_ids = torch.from_numpy(np.array([]))
        for i in range(num_classes):
            candidate_ids = torch.from_numpy(np.setdiff1d(all_ids, chosen_ids))
            new_chosen_idx = self._find_loss_augmented_facility_idx(pairwise_distances,
                                                                    labels, chosen_ids,
                                                                    candidate_ids,
                                                                    margin_multiplier,
                                                                    margin_type)
            chosen_ids = chosen_ids.tolist()
            chosen_ids.append(new_chosen_idx)
            chosen_ids = torch.from_numpy(np.array(chosen_ids))

        return chosen_ids  # Tensor(dtype=int32) 1D [num_class,]

    def compute_facility_energy(self, pairwise_distances, centroid_ids):
        """Compute the average travel distance to the assigned centroid.

        Args:
          pairwise_distances: 2-D Tensor of pairwise distances.
          centroid_ids: 1-D Tensor of indices.

        Returns:
          facility_energy: dtypes.float32 scalar.
        """
        return -1.0 * torch.sum(torch.min(tl.gather(pairwise_distances, centroid_ids), 0)[0])

    def update_1d_tensor(self, y, index, value):
        """Updates 1d tensor y so that y[index] = value.

        Args:
          y: 1-D Tensor.
          index: index of y to modify.
          value: new value to write at y[index].

        Returns:
          y_mod: 1-D Tensor. Tensor y after the update.
        """
        # value = array_ops.squeeze(value)
        # modify the 1D tensor x at index with value.
        # ex) chosen_ids = update_1D_tensor(chosen_ids, cluster_idx, best_medoid)
        # y_before = array_ops.slice(y, [0], [index])
        # y_after = array_ops.slice(y, [index + 1], [-1])
        # y_mod = array_ops.concat([y_before, [value], y_after], 0)
        y[index] = value
        return y

    def update_medoid_per_cluster(self, pairwise_distances, pairwise_distances_subset,
                                  labels, chosen_ids, cluster_member_ids,
                                  cluster_idx, margin_multiplier, margin_type):
        """Updates the cluster medoid per cluster.

        Args:
          pairwise_distances: 2-D Tensor of pairwise distances.
          pairwise_distances_subset: 2-D Tensor of pairwise distances for one cluster.
          labels: 1-D Tensor of ground truth cluster assignment.
          chosen_ids: 1-D Tensor of cluster centroid indices.
          cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
          cluster_idx: Index of this one cluster.
          margin_multiplier: multiplication constant.
          margin_type: Type of structured margin to use. Default is nmi.

        Returns:
          chosen_ids: Updated 1-D Tensor of cluster centroid indices.
        """

        # pairwise_distances_subset is of size [p, 1, 1, p],
        #   the intermediate dummy dimensions at
        #   [1, 2] makes this code work in the edge case where p=1.
        #   this happens if the cluster size is one.
        scores_fac = -1.0 * torch.sum(pairwise_distances_subset, 0)
        num_candidates = len(cluster_member_ids)
        scores_margin = np.zeros([num_candidates])

        assert num_candidates != 0

        candidate_scores = scores_fac.detach().cpu().numpy()  # + margin_multiplier * scores_margin
        argmax_index = np.argmax(candidate_scores, axis=0)
        best_medoid = cluster_member_ids[argmax_index]
        chosen_ids = self.update_1d_tensor(chosen_ids, cluster_idx, best_medoid)

        return chosen_ids

    def update_all_medoids(self, pairwise_distances, predictions, labels, chosen_ids,
                           margin_multiplier, margin_type):
        """Updates all cluster medoids a cluster at a time.

        Args:
          pairwise_distances: 2-D numpy array of pairwise distances.
          predictions: 1-D numpy array of predicted cluster assignment.
          labels: 1-D numpy array of ground truth cluster assignment.
          chosen_ids: 1-D numpy array of cluster centroid indices.
          margin_multiplier: multiplication constant.
          margin_type: Type of structured margin to use. Default is nmi.

        Returns:
          chosen_ids: Updated 1-D numpy array of cluster centroid indices.
        """
        unique_class_ids = np.unique(labels.cpu())
        num_classes = len(unique_class_ids)

        for i in range(num_classes):
            mask = torch.from_numpy(np.equal(predictions.cpu().numpy(), i))
            this_cluster_ids = torch.where(mask)[0]
            temp = (tl.gather(pairwise_distances, this_cluster_ids)).T
            pairwise_distances_subset = (tl.gather(temp, this_cluster_ids)).T
            chosen_ids = self.update_medoid_per_cluster(pairwise_distances,
                                                        pairwise_distances_subset, labels,
                                                        chosen_ids, this_cluster_ids,
                                                        i, margin_multiplier,
                                                        margin_type)

        return chosen_ids

    def compute_augmented_facility_locations_pam(self, pairwise_distances,
                                                 labels,
                                                 margin_multiplier,
                                                 margin_type,
                                                 chosen_ids,
                                                 pam_max_iter=5):
        """Refine the cluster centroids with PAM local search.

        For fixed iterations, alternate between updating the cluster assignment
          and updating cluster medoids.

        Args:
          pairwise_distances: 2-D Tensor of pairwise distances.
          labels: 1-D Tensor of ground truth cluster assignment.
          margin_multiplier: multiplication constant.
          margin_type: Type of structured margin to use. Default is nmi.
          chosen_ids: 1-D Tensor of initial estimate of cluster centroids.
          pam_max_iter: Number of refinement iterations.

        Returns:
          chosen_ids: Updated 1-D Tensor of cluster centroid indices.
        """
        for _ in range(pam_max_iter):
            # update the cluster assignment given the chosen_ids (S_pred)
            predictions = self.get_cluster_assignment(pairwise_distances, chosen_ids)

            # update the medoids per each cluster
            chosen_ids = self.update_all_medoids(pairwise_distances, predictions, labels,
                                                 chosen_ids, margin_multiplier, margin_type)

        return chosen_ids

    def compute(self, pairwise_distances, labels, opt: Option, global_step):
        """Computes the clustering loss.

        The following structured margins are supported:
          nmi: normalized mutual information
          ami: adjusted mutual information
          ari: adjusted random index
          vmeasure: v-measure
          const: indicator checking whether the two clusterings are the same.

        Args:
          labels: 1-D Tensor of labels of shape [batch size, ]
          feature: 2-D Tensor of embeddings of shape
            [batch size, embedding dimension]. Embeddings should be l2 normalized.
          margin_multiplier: float32 scalar. multiplier on the structured margin term
            See section 3.2 of paper for discussion.
          enable_pam_finetuning: Boolean, Whether to run local pam refinement.
            See section 3.4 of paper for discussion.
          margin_type: Type of structured margin to use. See section 3.2 of
            paper for discussion. Can be 'nmi', 'ami', 'ari', 'vmeasure', 'const'.
          print_losses: Boolean. Option to print the loss.

        Paper: https://arxiv.org/abs/1612.01213.

        Returns:
          clustering_loss: A float32 scalar `Tensor`.
        Raises:
          ImportError: If sklearn dependency is not installed.
        """

        margin_multiplier = opt.margin_multiplier
        enable_pam_finetuning = opt.enable_pam_finetuning,
        margin_type = opt.margin_type
        pam_max_iter = opt.pam_max_iter

        if opt.margin_step is not None:
            margin_multiplier = opt.margin_multiplier * pow(opt.multiplier_decay_ratio, global_step / opt.margin_step)

        all_ids = np.arange(pairwise_distances.shape[0])  # 1D numpy array

        # Compute the loss augmented inference and get the cluster centroids.
        chosen_ids = self.compute_augmented_facility_locations(pairwise_distances, labels, all_ids, margin_multiplier,
                                                               margin_type)
        # chosen_ids ：1D Tensor [num_class,]
        score_pred = self.compute_facility_energy(pairwise_distances, chosen_ids)

        # Given the predicted centroids, compute the cluster assignments.
        predictions = self.get_cluster_assignment(pairwise_distances, chosen_ids)
        clustering_score_pred = self.compute_clustering_score(labels.cpu(), predictions.cpu(),
                                                              margin_type)
        # Branch whether to use PAM finetuning.
        if enable_pam_finetuning:
            # Initialize with augmented facility solution.
            chosen_ids_pam = self.compute_augmented_facility_locations_pam(pairwise_distances,
                                                                           labels,
                                                                           margin_multiplier,
                                                                           margin_type,
                                                                           copy.deepcopy(chosen_ids),
                                                                           pam_max_iter)

            score_pred_pam = self.compute_facility_energy(pairwise_distances, chosen_ids_pam)
            predictions_pam = self.get_cluster_assignment(pairwise_distances, chosen_ids_pam)
            clustering_score_pred_pam = self.compute_clustering_score(labels.cpu(), predictions_pam.cpu(),
                                                                      margin_type)

            if score_pred_pam > score_pred:
                chosen_ids = chosen_ids_pam
                score_pred = score_pred_pam
                predictions = predictions_pam
                clustering_score_pred = clustering_score_pred_pam

                opt.pam_good += 1

        # Compute the clustering score from labels.
        score_gt = self.compute_gt_cluster_score(pairwise_distances, labels)

        # Compute the hinge loss.
        clustering_loss = max(score_pred + margin_multiplier * (1.0 - clustering_score_pred) - score_gt, 0.0)

        if score_pred - score_gt < 0.0:
            record = {"score_pred": score_pred.item(), "score_gt": score_gt.item(),
                      "margin": margin_multiplier * (1.0 - clustering_score_pred)}
            opt.zeros_record.append(record)
            if score_pred + margin_multiplier * (1.0 - clustering_score_pred) - score_gt > 0.0:
                opt.rescue += 1

        return clustering_loss
