from .cider_scorer import CiderScorer
import pdb

'''
main Class to compute the CIDEr metric
'''
class Cider:

    def __init__(self, test = None, refs = None, n = 4, sigma = 6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    '''
    main function to compute CIDEr score
    
    input params:
        reference(list): reference sentences ([[ref1a, ref1b, ref1c], ..., [refna, refnb]])
        hypothesis(list): predicted sentences ([[hypo1], [hypo2], ..., [hypon]])
    return: 
        cider(float): computed CIDEr score for the corpus
    '''
    def compute_score(self, reference, hypothesis):

        assert len(reference) == len(hypothesis)

        cider_scorer = CiderScorer(n = self._n, sigma = self._sigma)

        for id, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[id]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"