import distutils.util

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(label, pred, positive_label=1):
    """
    Python compute equal error rate (eer)
    ONLY tested on binary classification

    :param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
    :param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
    :param positive_label: the class that is viewed as positive class when computing EER
    :return: equal error rate (EER)
    """

    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)
