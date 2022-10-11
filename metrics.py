import numpy as np


class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: target
        # axis = 1: prediction
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            
            self.confusion_matrix += tmp

    def get_scores(self):
        hist = self.confusion_matrix
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / (union + 1e-7)
        mean_iou = np.mean(np.nan_to_num(iou[1:]))

        return {
                'iou': iou, 
                'iou_mean': mean_iou, 
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))