import numpy as np

def cal_(preds, trues, threshold):
    preds = np.nan_to_num(preds.copy(), nan=0.0)
    trues = np.nan_to_num(trues.copy(), nan=0.0)
    
    ## TP(true positive)
    TP = np.sum((preds >= threshold) & (trues >= threshold))

    ## TN(true negative)
    TN = np.sum((preds < threshold) & (trues < threshold))

    ## FP(false positive)
    FP = np.sum((preds >= threshold) & (trues < threshold))
    
    ## FN(false negative)
    FN = np.sum((preds < threshold) & (trues >= threshold))

    return TP, TN, FP, FN

def get_CSI(preds, trues, threshold=0.1):
    
    TP, TN, FP, FN = cal_(preds, trues, threshold)

    return TP / (TP + FP + FN)


def get_CSI_along_time(preds, trues, threshold=0.1):
    """
    dims -> (samples, frames, width, height)
    """

    all_ts = []
    for t in range(preds.shape[1]):
        threat_score = get_CSI(preds[:, t, :, :], trues[:, t, :, :], threshold)
        all_ts.append(threat_score)

    return all_ts
