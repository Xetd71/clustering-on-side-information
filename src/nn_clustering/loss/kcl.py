import torch.nn as nn
from helpers import DefaultValues


class KLDiv(nn.Module):
    # Calculate KL-Divergence

    def forward(self, predict, target):
        assert predict.ndimension() == 2, 'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict += DefaultValues.EPS
        target += DefaultValues.EPS
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld


class KCL(nn.Module):
    # KLD-based Clustering Loss (KCL)

    def __init__(self, margin=2.0):
        super(KCL,self).__init__()
        self.kld = KLDiv()
        self.hingeloss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert (
            len(prob1) == len(prob2) == len(simi),
            'Wrong input size:{0},{1},{2}'.format(str(len(prob1)), str(len(prob2)), str(len(simi)))
        )

        kld = self.kld(prob1, prob2)
        output = self.hingeloss(kld, simi)
        return output