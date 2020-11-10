# -*- coding: <encoding name> -*- : # -*- coding: utf-8 -*-···
import argparse
import warnings
import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorboardX import SummaryWriter

import config
from utils import dataloader, dataloader_BERT
from utils import tool_box as tl
from utils import messager
from utils.Evaluation import ClusterEvaluation
from modules.module.BasicModule import BasicModel
from modules import more
from modules.Cluster import K_means, K_means_BERT, mean_shift, mean_shift_BERT


def k_means_cluster_evaluation(model: BasicModel, opt: config.Option, test_data, test_data_label, loger):
    print(">>> last step model test")
    with torch.no_grad():
        cluster_result, cluster_msg, cluster_center, features = K_means_BERT(test_data, model.pred_vector,
                                                                             test_data_label,
                                                                             opt) if opt.BERT else K_means(test_data,
                                                                                                        model.pred_vector,
                                                                                                        len(np.unique(
                                                                                                        test_data_label)),
                                                                                                        opt)
        cluster_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
            print_flag=opt.print_losses, extra_info=True)
        NMI_scores_last = normalized_mutual_info_score(test_data_label, cluster_result)

        print("last step b3:", cluster_b3)
        print("last step NMI:", NMI_scores_last)

    model.save_last_step(opt.save_model_name)
    model.load_model(opt.save_model_name + "_best.pt")
    with torch.no_grad():
        cluster_result, cluster_msg, cluster_center, features = K_means_BERT(test_data, model.pred_vector,
                                                                             test_data_label,
                                                                             opt) if opt.BERT else K_means(test_data,
                                                                                                       model.pred_vector,
                                                                                                       len(np.unique(
                                                                                                       test_data_label)),
                                                                                                       opt)
        cluster_eval_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
            print_flag=opt.print_losses, extra_info=True)

        NMI_score = normalized_mutual_info_score(test_data_label, cluster_result)

        if opt.whether_visualize:
            loger.add_embedding(features, metadata=test_data_label, label_img=None,
                                global_step=0, tag='test_ground_truth',
                                metadata_header=None)
            loger.add_embedding(features, metadata=cluster_result, label_img=None,
                                global_step=0, tag='test_prediction',
                                metadata_header=None)
            loger.add_scalar('test_NMI', NMI_score, global_step=0)
            loger.add_scalar('test_F1', cluster_eval_b3['F1'], global_step=0)
            loger.add_scalar('test_precision', cluster_eval_b3['precision'],
                             global_step=0)
            loger.add_scalar('test_recall', cluster_eval_b3['recall'],
                             global_step=0)

    best_test_b3 = cluster_eval_b3

    return best_test_b3, NMI_score


def process_and_train_FL(model: BasicModel, opt: config.Option):
    # preparing saving files.
    save_path = os.path.join(opt.save_dir + '/model_file', opt.save_model_name).replace('\\', '/')
    print("model file save path: ", save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,
                     types=['train_data_file', 'val_data_file', 'test_data_file', 'load_model_name', 'save_model_name',
                            'trainset_loss_type', 'testset_loss_type',
                            'class_num_ratio'],
                     json_name='train_information_msg_' + time.strftime('%m{}%d{}_%H:%M'.format('月', '日')) + '.json')
    msger.record_message(
        [opt.train_data_file, opt.val_data_file, opt.test_data_file, opt.load_model_name, opt.save_model_name,
         opt.train_loss_type, opt.testset_loss_type,
         opt.class_num_ratio])
    msger.save_json()

    # train data loading
    print('-----Data Loading-----')
    if opt.BERT:
        dataloader_train = dataloader_BERT(opt.train_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                           opt.same_level_pair_file,
                                           max_len=opt.max_len, random_init=opt.random_init, seed=opt.seed)
        dataloader_val = dataloader_BERT(opt.val_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                         max_len=opt.max_len)
        dataloader_test = dataloader_BERT(opt.test_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                          max_len=opt.max_len)
    else:
        dataloader_train = dataloader(opt.train_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                      opt.same_level_pair_file,
                                      max_len=opt.max_len, random_init=opt.random_init, seed=opt.seed,
                                      data_type=opt.data_type)
        dataloader_val = dataloader(opt.val_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                    max_len=opt.max_len, data_type=opt.data_type)
        dataloader_test = dataloader(opt.test_data_file, opt.wordvec_file, opt.rel2id_file, opt.similarity_file,
                                     max_len=opt.max_len, data_type=opt.data_type)
    word_emb_dim = dataloader_train._word_emb_dim_()
    word_vec_mat = dataloader_train._word_vec_mat_()  # numpy.array float32
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Initializing-----')
    if opt.BERT != True:
        model.set_embedding_weight(word_vec_mat)

    if opt.load_model_name is not None:
        model.load_model(opt.load_model_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if torch.cuda.is_available():
        torch.cuda.set_device(int(opt.gpu))

    model, cuda_flag = tl.cudafy(model)
    if not cuda_flag:
        print("There is no gpu,use default cpu")

    count = tl.count_parameters(model)
    print("num of parameters:", count)

    # if the datasets are imbalanced such as nyt_su or trex , we load all test/dev data to perform open setting
    print('-----Validation Data Preparing-----')
    try:
        opt.data_type.index('imbalance')
        print("try load all imbalance dev data!")
        val_data, val_data_label = dataloader_val._data_()
    except:
        print("load part of data")
        if opt.data_type.startswith('fewrel'):
            val_data, val_data_label = dataloader_val._part_data_(
                100)  # 16 relation classes in validation data,each class has 100 sample in fewrel
        else:
            # other data sets has the problem of label imbalance
            val_data, val_data_label = dataloader_val._part_data_(
                100)  # for nyt_fb :sample 5 instance per relation, will get 490 dev instance

    print("-------Test Data Preparing--------")
    try:
        opt.data_type.index('imbalance')
        print("try load all imbalance test data!")
        test_data, test_data_label = dataloader_test._data_()
    except:
        print("load part of data")
        if opt.data_type.startswith('fewrel'):
            test_data, test_data_label = dataloader_test._data_()
        else:
            test_data, test_data_label = dataloader_test._data_(100)  # sample as the dev setting

    print("val_data:", len(val_data))
    print("val_data_label:", len(set(val_data_label)))
    print("test_data:", len(test_data))
    print("test_data_label:", len(set(test_data_label)))

    # intializing parameters
    batch_num_list = opt.batch_num

    msger_cluster = messager(save_path=save_path,
                             types=['method', 'temp_epoch', 'temp_batch_num', 'temp_batch_size', 'temp_lr', 'NMI', 'F1',
                                    'precision', 'recall', 'msg'],
                             json_name='Validation_cluster_msg_' + time.strftime(
                                 '%m{}%d{}_%H:%M'.format('月', '日')) + '.json')

    if opt.record_test:
        msger_test = messager(save_path=save_path,
                              types=['temp_globle_step', 'temp_batch_size', 'temp_learning_rate', 'NMI', 'F1',
                                     'precision', 'recall', 'msg'],
                              json_name='Test_cluster_msg_' + time.strftime(
                                  '%m{}%d{}_%H:%M'.format('月', '日')) + '.json')

    if opt.whether_visualize:
        loger = SummaryWriter(comment=opt.save_model_name)
    else:
        loger = None

    best_batch_step = 0
    best_epoch = 0
    batch_size_chose = -1
    print_flag = opt.print_losses
    best_validation_f1 = 0
    best_test_f1 = 0
    loss_list = []
    global_step = 0
    for epoch in range(opt.epoch_num):
        print('------epoch {}------'.format(epoch))
        print('max batch num to train is {}'.format(batch_num_list[epoch]))
        loss_reduce = 10000.
        early_stop_record = 0
        for i in range(1, batch_num_list[epoch] + 1):
            global_step += 1
            loss_list = model.train_self(opt, dataloader_train, loss_list, loger,
                                         batch_chose=batch_size_chose,
                                         global_step=global_step, temp_epoch=epoch)

            # print loss & record loss
            if i % 100 == 0:
                ave_loss = sum(loss_list) / 100.
                print('temp_batch_num: ', i, ' total_batch_num: ', batch_num_list[epoch], " ave_loss: ", ave_loss,
                      ' temp learning rate: ', opt.lr[opt.lr_chose])
                # empty the loss list
                loss_list = []
                # visualize
                if opt.whether_visualize:
                    loger.add_scalar('all_epoch_loss', ave_loss, global_step=global_step)
                # early stop
                if opt.early_stop is not None:
                    if ave_loss < loss_reduce:
                        early_stop_record = 0
                        loss_reduce = ave_loss
                    else:
                        early_stop_record += 1
                    if early_stop_record == opt.early_stop:
                        print("~~~~~~~~~ The loss can't be reduced in {} step, early stop! ~~~~~~~~~~~~".format(
                            opt.early_stop * 100))
                        cluster_result, cluster_msg, cluster_center, features = K_means_BERT(test_data,
                                                                                             model.pred_vector,
                                                                                             test_data_label,
                                                                                             opt) if opt.BERT else K_means(
                            test_data, model.pred_vector, len(np.unique(test_data_label)), opt)
                        cluster_test_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
                            extra_info=True, print_flag=True)
                        print("learning rate decay num:", opt.lr_decay_num)
                        print("learning rate decay step:", opt.lr_decay_record)
                        print("best_epoch:", best_epoch)
                        print("best_step:", best_batch_step)
                        print("best_batch_size:", best_batch_size)
                        print("best_cluster_eval_b3:", best_validation_f1)
                        print("seed:", opt.seed)

            # clustering & validation
            if i % 200 == 0:
                print(opt.save_model_name, 'epoch:', epoch)
                with torch.no_grad():
                    # fewrel -> K-means ; nyt+su -> Mean-Shift
                    if opt.dataset.startswith("fewrel"):
                        print("chose k-means >>>")
                        F_score = -1.0
                        best_cluster_result = None
                        best_cluster_msg = None
                        best_cluster_center = None
                        best_features = None
                        best_cluster_eval_b3 = None
                        for iterion in range(opt.eval_num):
                            K_num = opt.K_num if opt.K_num != 0 else len(np.unique(val_data_label))
                            cluster_result, cluster_msg, cluster_center, features = K_means_BERT(val_data,
                                                                                                 model.pred_vector,
                                                                                                 val_data_label,
                                                                                                 opt) if opt.BERT else K_means(
                                val_data, model.pred_vector, K_num, opt)

                            cluster_eval_b3 = ClusterEvaluation(val_data_label, cluster_result).printEvaluation(
                                print_flag=False)

                            if F_score < cluster_eval_b3['F1']:
                                F_score = cluster_eval_b3['F1']
                                best_cluster_result = cluster_result
                                best_cluster_msg = cluster_msg
                                best_cluster_center = cluster_center
                                best_features = features
                                best_cluster_eval_b3 = cluster_eval_b3

                        cluster_result = best_cluster_result
                        cluster_msg = best_cluster_msg
                        cluster_center = best_cluster_center
                        features = best_features
                        cluster_eval_b3 = best_cluster_eval_b3

                    else:
                        print("chose mean-shift >>>")
                        cluster_result, cluster_msg, cluster_center, features = mean_shift_BERT(val_data,
                                                                                    model.pred_vector,
                                                                                    val_data_label,
                                                                                    opt) if opt.BERT else mean_shift(
                                                                                    val_data, model.pred_vector,opt)

                        cluster_eval_b3 = ClusterEvaluation(val_data_label, cluster_result).printEvaluation(
                            print_flag=False, extra_info=True)

                    NMI_score = normalized_mutual_info_score(val_data_label, cluster_result)

                    print("NMI:{} ,F1:{} ,precision:{} ,recall:{}".format(NMI_score, cluster_eval_b3['F1'],
                                                                          cluster_eval_b3['precision'],
                                                                          cluster_eval_b3['recall'], ))

                    msger_cluster.record_message(
                        [opt.select_cluster, epoch, i, opt.batch_size[batch_size_chose], model.lr, NMI_score,
                         cluster_eval_b3['F1'], cluster_eval_b3['precision'],
                         cluster_eval_b3['recall'], cluster_msg])

                    msger_cluster.save_json()

                    two_f1 = cluster_eval_b3['F1']
                    if two_f1 > best_validation_f1:  # acc
                        if opt.record_test == False:
                            model.save_model(model_name=opt.save_model_name, global_step=global_step)
                        best_batch_step = i
                        best_epoch = epoch
                        best_batch_size = opt.batch_size[batch_size_chose]
                        best_validation_f1 = two_f1

                    if opt.whether_visualize:
                        loger.add_embedding(features, metadata=val_data_label, label_img=None,
                                            global_step=global_step, tag='ground_truth',
                                            metadata_header=None)
                        loger.add_embedding(features, metadata=cluster_result, label_img=None,
                                            global_step=global_step, tag='prediction',
                                            metadata_header=None)
                        loger.add_scalar('all_epoch_NMI', NMI_score, global_step=global_step)
                        loger.add_scalar('all_epoch_F1', cluster_eval_b3['F1'], global_step=global_step)
                        loger.add_scalar('all_epoch_precision', cluster_eval_b3['precision'],
                                         global_step=global_step)
                        loger.add_scalar('all_epoch_recall', cluster_eval_b3['recall'],
                                         global_step=global_step)

                    if opt.record_test:
                        if opt.dataset.startswith("fewrel"):
                            cluster_result, cluster_msg, cluster_center, features = K_means_BERT(test_data,
                                                                                                 model.pred_vector,
                                                                                                 test_data_label,
                                                                                                 opt) if opt.BERT else K_means(
                                test_data, model.pred_vector, len(np.unique(test_data_label)), opt)

                            cluster_test_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
                                print_flag=False)
                        else:
                            cluster_result, cluster_msg, cluster_center, features = mean_shift_BERT(test_data,
                                                                                            model.pred_vector,
                                                                                            test_data_label,
                                                                                            opt) if opt.BERT else mean_shift(
                                                                                            test_data, model.pred_vector,opt)

                            cluster_test_b3 = ClusterEvaluation(test_data_label, cluster_result).printEvaluation(
                                print_flag=False, extra_info=True)

                        msger_test.record_message(
                            [global_step, opt.batch_size[batch_size_chose], opt.lr[opt.lr_chose], NMI_score,
                             cluster_test_b3['F1'], cluster_test_b3['precision'],
                             cluster_test_b3['recall'], cluster_msg])
                        msger_test.save_json()
                        print('test messages saved.')

                        if cluster_test_b3['F1'] > best_test_f1:
                            model.save_model(model_name=opt.save_model_name, global_step=global_step)
                            best_batch_step = i
                            best_epoch = epoch
                            best_batch_size = opt.batch_size[batch_size_chose]
                            best_test_f1 = cluster_test_b3['F1']

        model.lr_decay(opt)
        opt.lr_decay_record.append(global_step)

        print('End: The model is:', opt.save_model_name, opt.train_loss_type, opt.testset_loss_type)

    if opt.dataset.startswith("fewrel"):
        print('\n-----K-means Clustering test-----')
        best_test_b3, NMI_score = k_means_cluster_evaluation(model, opt, test_data, test_data_label, loger)
    else:
        print("\n-----------Mean_shift Clustering test:---------------")
        model.load_model(opt.save_model_name + "_best.pt")
        cluster_result_ms, cluster_msg_ms, _, _ = mean_shift_BERT(test_data, model.pred_vector, test_data_label,
                                                                  opt) if opt.BERT else mean_shift(test_data,
                                                                                                   model.pred_vector,
                                                                                                   opt)
        cluster_eval_b3_ms = ClusterEvaluation(test_data_label, cluster_result_ms).printEvaluation(
            print_flag=opt.print_losses, extra_info=True)
        NMI_score_ms = normalized_mutual_info_score(test_data_label, cluster_result_ms)

        best_test_b3 = cluster_eval_b3_ms
        NMI_score = NMI_score_ms

        if opt.whether_visualize:
            loger.add_scalar('test_NMI_MeanShift', NMI_score_ms, global_step=0)
            loger.add_scalar('test_F1_MeanShift', cluster_eval_b3_ms['F1'], global_step=0)

    print("learning rate decay num:", opt.lr_decay_num)
    print("learning rate decay step:", opt.lr_decay_record)
    print("best_epoch:", best_epoch)
    print("best_step:", best_batch_step)
    print("best_batch_size:", best_batch_size)
    print("best_cluster_eval_b3:", best_validation_f1)
    print("best_cluster_test_b3:", best_test_b3)
    print("best_NMI_score:", NMI_score)
    print("seed:", opt.seed)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--dataset", type=str, default='fewrel')
    parser.add_argument("--train_loss_type", type=str, default="Rank List Loss")
    parser.add_argument("--testset_loss_type", type=str, default='none')

    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--learning_rate_linear", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lr_chose", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=240)
    parser.add_argument("--batch_num", type=int, default=2000)
    parser.add_argument("--epoch_num", type=int, default=4)
    parser.add_argument("--class_ratio", type=float, default=0.1)
    parser.add_argument("--squared", type=int, default=0)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--record_test", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--pos_emb_dim", type=int, default=10)
    parser.add_argument("--drop_out", type=float, default=0.2)
    # clustering
    parser.add_argument("--K_num", type=int, default=0)
    parser.add_argument("--band", type=float, default=0.7784975910442384)
    # RLL
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--alpha_rank", type=float, default=1.2)
    parser.add_argument("--temp_neg", type=int, default=10)
    parser.add_argument("--temp_pos", type=int, default=0)
    parser.add_argument("--landa", type=float, default=0.5)
    # VAT
    parser.add_argument("--VAT", type=float, default=0)
    parser.add_argument("--power_iterations", type=int, default=1)
    parser.add_argument("--p_mult", type=float, default=0.02)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--lambda_V", type=float, default=1)
    # self
    parser.add_argument("--inclass_augment", type=int, default=0)
    parser.add_argument("--margin_un", type=float, default=0.4)
    parser.add_argument("--alpha_rank_un", type=float, default=1.2)
    parser.add_argument("--uninfor_landa", type=float, default=0.0)
    parser.add_argument("--uninfor_drop", type=float, default=0.0)
    # point
    parser.add_argument("--save_model_name", type=str, default='RRL_test')
    parser.add_argument("--load_model_name", type=str, default=None)

    parser.add_argument("--whether_visualize", type=int, default=1)

    # BERT
    parser.add_argument("--BERT", type=int, default=0)

    args = parser.parse_args()
    args.whether_visualize = True if args.whether_visualize == 1 else False
    args.inclass_augment = True if args.inclass_augment == 1 else False
    args.squared = True if args.squared == 1 else False
    args.record_test = True if args.record_test == 1 else False
    args.BERT = True if args.BERT == 1 else False

    opt = config.Bert_option(args.gpu) if args.BERT else config.Option()
    opt.dataset = args.dataset
    if args.dataset.startswith("fewrel"):
        opt.data_type = 'fewrel_ori'
        opt.train_data_file = './data/datasets/fewrel_ori/fewrel80_train.json'
        opt.val_data_file = './data/datasets/fewrel_ori/fewrel80_test_train.json'
        opt.test_data_file = './data/datasets/fewrel_ori/fewrel80_test_test.json'
        opt.wordvec_file = "./data/wordvec/word_vec.json"
        opt.rel2id_file = './data/support_files/fewrel_rel2id.json'
    elif args.dataset.startswith("nyt"):
        opt.data_type = 'nyt_su_imbalance'
        opt.train_data_file = './data/datasets/nyt_su/nyt_supervision_train.json'
        opt.val_data_file = './data/datasets/nyt_su/nyt_supervision_dev.json'
        opt.test_data_file = './data/datasets/nyt_su/nyt_supervision_test.json'
        opt.wordvec_file = "./data/wordvec/word_vec.json"
        opt.rel2id_file = './data/support_files/nyt_su_rel2id.json'
    else:
        raise NameError("please use fewrel or nyt+su datasets!")

    opt.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(opt.gpu))
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0), " totally {} GPU".format(n_gpu), " use cuda:",
          os.environ['CUDA_VISIBLE_DEVICES'])

    # initialize option
    opt.train_loss_type = args.train_loss_type
    opt.testset_loss_type = args.testset_loss_type

    opt.load_model_name = args.load_model_name  # e.g. 'MORE_best.pt'
    opt.save_model_name = args.save_model_name

    opt.epoch_num = args.epoch_num
    opt.lr = [args.learning_rate] * opt.epoch_num
    opt.lr_linear = args.learning_rate_linear
    opt.lr_chose = args.lr_chose
    opt.batch_size = [args.batch_size] * opt.epoch_num
    opt.class_num_ratio = [args.class_ratio] * opt.epoch_num
    opt.batch_num = [args.batch_num] * opt.epoch_num
    opt.squared = args.squared
    opt.alpha = args.alpha
    opt.seed = args.seed
    opt.early_stop = args.early_stop  # None，if no early stop
    opt.record_test = args.record_test
    opt.whether_visualize = args.whether_visualize
    opt.embedding_dim = args.embedding_dim
    opt.max_len = args.max_len
    opt.pos_emb_dim = args.pos_emb_dim
    opt.drop_out = args.drop_out
    opt.K_num = args.K_num
    opt.band = args.band
    opt.select_cluster = "K-means" if opt.dataset.startswith("fewrel") else 'mean_shift'
    opt.margin = args.margin
    opt.margin_un = args.margin_un
    opt.alpha_rank = args.alpha_rank
    opt.alpha_rank_un = args.alpha_rank_un
    opt.temp_neg = args.temp_neg
    opt.temp_pos = args.temp_pos
    opt.landa = args.landa
    opt.inclass_augment = args.inclass_augment
    opt.VAT = args.VAT
    opt.p_mult = args.p_mult
    opt.power_iterations = args.power_iterations
    opt.warm_up = args.warm_up
    opt.lambda_V = args.lambda_V

    opt.uninfor_landa = args.uninfor_landa
    opt.uninfor_drop = args.uninfor_drop

    random_init = np.random.randn(114042, 50)
    model = more.MORE_BERT(opt) if args.BERT else more.MORE(random_init, opt)

    print("-----The main hyparameters：----------\n")
    print("dataset: ", opt.data_type)
    print("epoch_num: ", opt.epoch_num, "\nlearning rate: ", opt.lr, "\nbatch_size: ", opt.batch_size,
          "\nclass_ratio: ", opt.class_num_ratio, "\nbatch_num: ", opt.batch_num)
    print("squared: ", opt.squared)
    print("margin: ", opt.margin, "\nalpha: ", opt.alpha_rank)
    VAT_FLAG = "Yes" if opt.VAT != 0 else "No"
    print("Whether add virtual adversarial training? ", VAT_FLAG)
    print("\n--------------------------------------\n")

    start_time = time.time()
    process_and_train_FL(model, opt)
    end_time = time.time()
    print("Running time:", (end_time - start_time) / 60, " mins")
