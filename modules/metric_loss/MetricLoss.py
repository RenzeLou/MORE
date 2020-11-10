import os
import torch
from utils import tool_box as tl
from config import Option
from modules.metric_loss import RankListLoss, SiameseLoss, FacilityLoss


def pairwise_distance(embeddings, squared=False):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    if pairwise_distances.is_cuda:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(tl.cudafy(torch.ones([num_data]))[0])
    else:
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))

    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances


class metric_loss(object):
    def __init__(self):
        super(metric_loss, self).__init__()

    def cluster_loss(self, opt: Option, feature, labels, global_step):

        assert len(labels.size()) == 1

        pairwise_distances = pairwise_distance(feature, opt.squared)  # [batch,batch]

        if opt.train_loss_type.startswith("Siamese"):
            criterion = SiameseLoss.siamese()
        elif opt.train_loss_type.startswith("Rank"):
            criterion = RankListLoss.rank_list()
        elif opt.train_loss_type.startswith("Facility"):
            criterion = FacilityLoss.Facility()

        clustering_loss = criterion.compute(pairwise_distances, labels, opt, global_step)

        return clustering_loss


if __name__ == "__main__":
    import warnings
    import time

    warnings.filterwarnings("ignore")
    print("hello")
    torch.save(torch.rand(3, 3), "/home/wyt/1_lrz/8-3/observe/step.pt")
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    ml = metric_loss()
    opt = Option()
    # f = tl.cudafy(Variable(torch.randn(128, 50),requires_grad=True))[0]
    # l=tl.cudafy(torch.cat((torch.ones([100]),torch.zeros([28])),0))[0]
    # torch.save(f,"/home/wyt/1_lrz/test/f.pt")
    # torch.save(l, "/home/wyt/1_lrz/test/l.pt")
    f = torch.load("/home/wyt/1_lrz/test/f.pt")
    l = torch.load("/home/wyt/1_lrz/test/l.pt")
    # ans = ml.pairwise_distance()
    # t1=time.time()
    loss, num_class = ml.cluster_loss(opt, f, l)
    # t2=time.time()
    print("loss:", loss)
    print("num_class:", num_class)
    # print("ttime:",t2-t1," seconds")
    # print(ans)
    # centroid_ids = torch.tensor([0, 2, 3])
    # pd = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [2, 3, 7, 6, 4], [2, 9, 7, 0, 4], [1, 8, 2, 3, 0]])
    # for i in range(5):
    #     pd[i][i] = 0
    # ml = metric_loss()
    # y = ml.get_cluster_assignment(pd, centroid_ids)
    # print(y)
