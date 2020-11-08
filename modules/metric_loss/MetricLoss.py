from modules.Cluster import euclidean_dist_rank
from config import Option
from modules.metric_loss import MetricLoss,RankListLoss,SiameseLoss,FacilityLoss


class metric_loss(object):
    def __init__(self):
        super(metric_loss, self).__init__()

    def cluster_loss(self, opt: Option, batch_data, feature, wordembed,labels, global_step,temp_epoch,forward_norm):

        assert len(labels.size()) == 1

        clustering_loss = None
        if opt.train_loss_type.startswith("Siamese"):
            pairwise_distances = euclidean_dist_rank(feature, feature)  # [batch,batch]
            criterion = SiameseLoss.siamese()
            clustering_loss = criterion.compute(pairwise_distances, labels, opt, global_step)
        elif opt.train_loss_type.startswith("Rank"):
            criterion = RankListLoss.rank_list()
            clustering_loss = criterion.compute(batch_data,feature, wordembed, labels, opt, temp_epoch,forward_norm)
        elif opt.train_loss_type.startswith("Facility"):
            pairwise_distances = euclidean_dist_rank(feature, feature)  # [batch,batch]
            criterion = FacilityLoss.Facility()
            clustering_loss = criterion.compute(pairwise_distances, labels, opt, global_step)

        if clustering_loss is None:
            raise Exception

        return clustering_loss

