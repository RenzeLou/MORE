import matplotlib.pyplot as plt
import numpy as np
import os
import time

def myplot(model_name, epoch, loss: list, batch_num_loss: list, score: dict = None, batch_num_score: list = None):
    save_path = '/home/wyt/lrz/RSN_torch_by_lrz/model_file'
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure()
    plt.plot(batch_num_loss, loss, color='r', label='ave_loss_100B', linewidth=2, markersize=10,
             alpha=0.8, mec='black', mew=0.7)
    plt.xlabel('batch_num', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.title("average loss")
    plt.legend(numpoints=2, fontsize='large')
    plt.grid(linestyle='--')
    plt.savefig(os.path.join(save_path, 'epoch_' + str(epoch) + '_loss_' + time.strftime(
        '%m{}%d{}%H{}%M{}'.format('月', '日','时','分')) + '.pdf'), format='pdf', bbox_inches='tight')
    # plt.show()
    if score is not None:
        plt.figure()
        color_dict = {"NMI": "green", "F1": "orange", "precision": "blue", "recall": "yellow"}
        for name, lis in score.items():
            lis = list(map(lambda x: 100 * x, lis))
            plt.plot(batch_num_score, lis, color=color_dict[name], label=name, linewidth=2, markersize=10,
                     alpha=0.8, mec='black', mew=0.7)
        plt.xlabel('batch_num')
        plt.ylabel('score(%)')
        plt.title("clustering score")
        plt.legend(numpoints=2, fontsize='large')
        plt.grid(linestyle='--')
        plt.savefig(os.path.join(save_path, 'epoch_' + str(epoch) + '_score_' + time.strftime(
            '%m{}%d{}%H{}%M{}'.format('月', '日','时','分')) + '.pdf'), format='pdf', bbox_inches='tight')

if __name__=="__main__":
    score={"NMI":np.random.rand(4).tolist(),
           "F1":np.random.rand(4).tolist(),
           "precision":np.random.rand(4).tolist(),
           "recall":np.random.rand(4).tolist()}
    myplot('test_plot',1,[0.03,0.04,0.12,0.15,0.17,0.09,0.13,0.12],[100,200,300,400,500,600,700,800],score,[200,400,600,800])