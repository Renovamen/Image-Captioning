#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from typing import List, Tuple
import numpy as np

from .bleu_scorer import BleuScorer

class Bleu:
    """Compute BLEU score for a set of candidate sentences."""

    def __init__(self, n: int = 4) -> None:
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(
        self, reference: List[List[str]], hypothesis: List[List[str]]
    ) -> Tuple[List[float], List[List[float]]]:
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
        average_score : List[float]
            Mean BLEU-1 to BLEU-4 score computed by averaging scores for all the images

        scores : List[List[float]]
            BLEU-1 to BLEU-4 scores computed for each image
        """
        assert len(reference) == len(hypothesis)

        bleu_scorer = BleuScorer(n = self._n)

        for id, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[id]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        return score, scores

    def method(self) -> str:
        return "Bleu"
