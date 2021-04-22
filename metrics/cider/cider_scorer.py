# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
from collections import defaultdict
from typing import Tuple, List
import numpy as np
import pdb
import math

def precook(s: str, n: int = 4, out: bool = False) -> dict:
    """
    Takes a string as input and returns an object that can be given to either
    cook_refs or cook_test. This is optional: cook_refs and cook_test can take
    string arguments as well.

    Parameters
    ----------
    s : str
        Sentence to be converted into ngrams

    n : int
        Number of ngrams for which representation is calculated

    Returns
    -------
    counts : dict
        Term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs: List[str], n: int = 4) -> List[dict]:  ## lhuang: oracle will call with "average"
    """
    Takes a list of reference sentences for a single segment and returns an
    object that encapsulates everything that BLEU needs to know about them.

    Parameters
    ----------
    refs : List[str]
        Reference sentences for some image

    n : int
        Number of ngrams for which (ngram) representation is calculated

    Returns
    -------
    result : List[dict]
    """
    return [precook(ref, n) for ref in refs]

def cook_test(test: str, n: int = 4) -> dict:
    """
    Takes a test sentence and returns an object that encapsulates everything that
    BLEU needs to know about it.

    Parameters
    ----------
    test : str
        Hypothesis sentence for an image

    n : int
        Number of ngrams for which (ngram) representation is calculated

    Returns
    -------
    result : dict
    """
    return precook(test, n, True)

class CiderScorer(object):
    """CIDEr scorer."""

    def copy(self):
        """Copy the refs."""
        new = CiderScorer(n = self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test = None, refs = None, n: int = 4, sigma: float = 6.0) -> None:
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        """Called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """Add an instance (e.g., from another sentence)."""
        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self) -> None:
        """
        Compute term frequency for reference data.

        This will be used to compute idf (inverse document frequency later).

        The term frequency is stored in the object.
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts) -> Tuple[List[dict], List[float], int]:
            """
            Function maps counts of ngram to vector of tfidf weights.

            The function returns vec, an array of dictionary that store mapping
            of n-gram and tf-idf weights.

            The n-th entry of array denotes length of n-grams.
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(
            vec_hyp: List[dict],
            vec_ref: List[dict],
            norm_hyp: List[float],
            norm_ref: List[float],
            length_hyp: int,
            length_ref: int
        ) -> np.ndarray:
            """
            Compute the cosine similarity of two vectors.

            Parameters
            ----------
            vec_hyp : List[dict]
                Array of dictionary for vector corresponding to hypothesis

            vec_ref : List[dict]
                Array of dictionary for vector corresponding to reference

            norm_hyp : List[float]
                Array of float for vector corresponding to hypothesis

            norm_ref : List[float]
                Array of float for vector corresponding to reference

            length_hyp : int
                Length of hypothesis

            length_ref : int
                Length of reference

            Returns
            -------
            similarity : np.ndarray
                Array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self) -> Tuple[np.float64, np.ndarray]:
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert(len(self.ctest) >= max(self.document_frequency.values()))
        # compute cider score
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)
