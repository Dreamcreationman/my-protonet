import os
import argparse


def visual_para(parser):
    print(format("   Args in   ", "=^55"))
    for k, v in vars(parser.parse_args()).items():
        print("||", format("{} : {}".format(k, v), " ^50"), "||")
    print(format("  Args parsed  ", "=^55"))

def get_parser():
    parser = argparse.ArgumentParser("== Define Parameters ==")

    # other
    parser.add_argument('-se', "--seed",
                        type=int,
                        default=5469,
                        help='input for the manual seeds initializations')

    parser.add_argument('-cuda', "--cuda",
                        type=bool,
                        default=True,
                        help="use gpu or not")

    # data in or out
    parser.add_argument('-dt', "--dataset_type",
                        type=str,
                        default="miniImagenet",
                        help="type of dataset")

    parser.add_argument('-root', "--dataset_root",
                        type=str,
                        default="/home/data/dataset/mini-imagenet",
                        help='root dir of dataset')

    parser.add_argument('-mo', '--model_out',
                        type=str,
                        default='..' + os.sep + 'model',
                        help='root where to store models, losses and accuracies')

    # hyperparameters about problem setting
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        default=100,
                        help='number of epochs for training')

    parser.add_argument('-eps', '--episodes',
                        type=int,
                        default=100,
                        help='number of episodes per epoch')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=1e-3,
                        help='learning rate for the model')

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        default=20,
                        help='StepLR learning rate scheduler step')

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        default=0.5,
                        help='StepLR learning rate scheduler gamma')

    # training
    parser.add_argument('-cTr', "--class_per_epi_tr",
                        type=int,
                        default=60,
                        help="number of random classes per episode for training")

    parser.add_argument('-nsTr', "--num_spt_tr",
                        type=int,
                        default=5,
                        help="number of samples per class to use as support for training")

    parser.add_argument('-nqTr', "--num_qry_tr",
                        type=int,
                        default=5,
                        help="nummer of samples per class to use as query for training")

    # validation
    parser.add_argument('-cVa', "--class_per_epi_val",
                        type=int,
                        default=5,
                        help="number of random classes per episode for validation")

    parser.add_argument('-nsVa', "--num_spt_val",
                        type=int,
                        default=5,
                        help="number of samples per class to use as support for validation")

    parser.add_argument('-nqVa', "--num_qry_val",
                        type=int,
                        default=15,
                        help="nummer of samples per class to use as query for validation")

    visual_para(parser)
    return parser


if __name__ == '__main__':
    get_parser()