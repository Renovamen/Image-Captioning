# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

from typing import List, Tuple
import numpy as np
import pdb

def my_lcs(string: List[str], sub: List[str]) -> int:
    """
    Calculates the longest common subsequence for a pair of tokenized strings.

    Parameters
    ----------
    string : List[str]
        Tokens from a string split using whitespace

    sub : List[str]
        Shorter string, also split using whitespace

    Returns
    -------
    length : int
        Length of the longest common subsequence between the two strings

    Notes
    -----
    ``my_lcs`` only gives length of the longest common subsequence, not the
    actual LCS.
    """
    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if(string[i - 1] == sub[j - 1]):
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j] , lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

class Rouge:
    """Compute ROUGE-L score for a set of candidate sentences"""

    def __init__(self) -> None:
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate: str, refs: List[str]) -> float:
        """
        Compute ROUGE-L score given one candidate and references for an image

        Parameters
        ----------
        candidate : str
            Candidate sentence to be evaluated

        refs : List[str]
            Reference sentences for the particular image to be evaluated

        Returns
        -------
        score : float
            ROUGE-L score for the candidate evaluated against references
        """
        # assert(len(candidate)==0)
        # assert(len(refs)>0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate.split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max != 0 and rec_max != 0):
            score = ((1 + self.beta**2) * prec_max * rec_max) / float(rec_max + self.beta**2 * prec_max)
        else:
            score = 0.0

        return score

    def compute_score(
        self, reference: List[List[str]], hypothesis: List[List[str]]
    ) -> Tuple[np.float64, np.ndarray]:
        """
        Compute Rouge-L score given a set of reference and candidate sentences
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
            Mean ROUGE-L score computed by averaging scores for all the images

        scores : np.ndarray
            ROUGE-L scores computed for each image
        """
        assert len(reference) == len(hypothesis)

        score = []
        for i, hypo in enumerate(hypothesis):
            hypo = hypo
            ref = reference[i]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            score.append(self.calc_score(hypo[0], ref))

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self) -> str:
        return "Rouge"
