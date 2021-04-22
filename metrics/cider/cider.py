# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#              by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from typing import List, Tuple
import numpy as np
import pdb

from .cider_scorer import CiderScorer

class Cider:
    """Compute CIDEr score for a set of candidate sentences."""

    def __init__(self, n: int = 4, sigma: float = 6.0) -> None:
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(
        self, reference: List[List[str]], hypothesis: List[List[str]]
    ) -> Tuple[np.float64, np.ndarray]:
        """
        Compute CIDEr score given a set of reference and candidate sentences
        for the dataset.

        Parameters
        ----------
        reference : List[List[str]] ([[ref1a, ref1b, ref1c], ..., [refna, refnb]])
            Reference sentences

        hypothesis : List[List[str]] ([[hypo1], [hypo2], ..., [hypon]])
            Predicted sentences

        Returns
        -------
        average_score : np.float64
            Mean CIDEr score computed by averaging scores for all the images

        scores : np.ndarray
            CIDEr scores computed for each image
        """

        assert len(reference) == len(hypothesis)

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for i, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[i]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            cider_scorer += (hypo[0], ref)

        score, scores = cider_scorer.compute_score()
        return score, scores

    def method(self) -> str:
        return "CIDEr"
