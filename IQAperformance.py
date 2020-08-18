from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats


class IQAPerformance(Metric):
    """
    Evaluation of VQA methods using SROCC, PLCC, RMSE.

    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, status='train', k=[1,1,1], b=[0,0,0], mapping=True):
        super(IQAPerformance, self).__init__()
        self.k = k
        self.b = b
        self.status = status
        self.mapping = mapping

    def reset(self):
        self._y_pred  = []
        self._y_pred1 = []
        self._y_pred2 = []
        self._y       = []
        self._y_std   = []

    def update(self, output):
        y_pred, y = output
        self._y.extend([t.item() for t in y[0]])
        self._y_std.extend([t.item() for t in y[1]])
        self._y_pred.extend([t.item() for t in y_pred[-1]])
        self._y_pred1.extend([t.item() for t in y_pred[0]])
        self._y_pred2.extend([t.item() for t in y_pred[1]])

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))

        pq_before = np.reshape(np.asarray(self._y_pred), (-1, 1))
        pq = self.linear_mapping(pq_before, sq, i=0)
        SROCC = stats.spearmanr(sq, pq)[0]
        PLCC = stats.pearsonr(sq, pq)[0]
        RMSE = np.sqrt(((sq - pq) ** 2).mean())
        # KROCC = stats.stats.kendalltau(sq, pq)[0]
        # outlier_ratio = (np.abs(sq - pq) > 2 * sq_std).mean()

        pq1_before = np.reshape(np.asarray(self._y_pred1), (-1, 1))
        pq2_before = np.reshape(np.asarray(self._y_pred2), (-1, 1))
        pq1 = self.linear_mapping(pq1_before, sq, i=1)
        pq2 = self.linear_mapping(pq2_before, sq, i=2)

        SROCC1 = stats.spearmanr(sq, pq1)[0]
        PLCC1 = stats.pearsonr(sq, pq1)[0]
        RMSE1 = np.sqrt(((sq - pq1) ** 2).mean())
        # KROCC1 = stats.stats.kendalltau(sq, pq1)[0]
        # outlier_ratio1 = (np.abs(sq - pq1) > 2 * sq_std).mean()

        SROCC2 = stats.spearmanr(sq, pq2)[0]
        PLCC2 = stats.pearsonr(sq, pq2)[0]
        RMSE2 = np.sqrt(((sq - pq2) ** 2).mean())
        # KROCC2 = stats.stats.kendalltau(sq, pq2)[0]
        # outlier_ratio2 = (np.abs(sq - pq2) > 2 * sq_std).mean()

        return {'SROCC': SROCC,
                'SROCC1': SROCC1,
                'SROCC2': SROCC2,
                'PLCC': PLCC,
                'PLCC1': PLCC1,
                'PLCC2': PLCC2,
                'RMSE': RMSE,
                'RMSE1': RMSE1,
                'RMSE2': RMSE2,
                'sq': sq,
                'sq_std': sq_std,
                'pq': pq,
                'pq1': pq1,
                'pq2': pq2,
                'pq_before': pq_before,
                'pq1_before': pq1_before,
                'pq2_before': pq2_before,
                'k': self.k,
                'b': self.b
                }

    def linear_mapping(self, pq, sq, i=0):
        if not self.mapping:
            return np.reshape(pq, (-1,))
        ones = np.ones_like(pq)
        yp1 = np.concatenate((pq, ones), axis=1)
        if self.status == 'train':
            # LSR solution of Q_i = k_1\hat{Q_i}+k_2. One can use the form of Eqn. (17) in the paper. 
            # However, for an efficient implementation, we use the matrix form of the solution here.
            # That is, h = (X^TX)^{-1}X^TY is the LSR solution of Y = Xh,
            # where X = [\hat{\mathbf{Q}}, \mathbf{1}], h = [k_1,k_2]^T, and Y=\mathbf{Q}.
            h = np.matmul(np.linalg.inv(np.matmul(yp1.transpose(), yp1)), np.matmul(yp1.transpose(), sq))
            self.k[i] = h[0].item()
            self.b[i] = h[1].item()
        else:
            h = np.reshape(np.asarray([self.k[i], self.b[i]]), (-1, 1))
        pq = np.matmul(yp1, h)

        return np.reshape(pq, (-1,))
