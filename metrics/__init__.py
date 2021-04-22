from typing import List, Tuple, Dict, Union
import numpy as np

from .bleu import Bleu
from .cider import Cider
from .meteor import Meteor
from .rouge import Rouge

class Metrics:
    """
    Compute metrics on given reference and candidate sentences set. Now supports
    BLEU, CIDEr, METEOR and ROUGE-L.

    Parameters
    ----------
    references : List[List[List[int]]] ([[[ref1a], [ref1b], [ref1c]], ..., [[refna], [refnb]]])
        Reference sencences (list of word ids)

    candidates : List[List[int]] ([[hyp1], [hyp2], ..., [hypn]]):
        Candidate sencences (list of word ids)

    rev_word_map : Dict[int, str]
        ix2word map
    """

    def __init__(
        self,
        references: List[List[List[int]]],
        candidates: List[List[int]],
        rev_word_map: Dict[int, str]
    ) -> None:
        corpus = setup_corpus(references, candidates, rev_word_map)
        self.ref_sentence = corpus[0]
        self.hypo_sentence = corpus[1]

    @property
    def belu(self) -> Tuple[float, float, float, float]:
        bleu_score = Bleu().compute_score(self.ref_sentence, self.hypo_sentence)
        return bleu_score[0][0], bleu_score[0][0], bleu_score[0][2], bleu_score[0][3]

    @property
    def cider(self) -> np.float64:
        cider_score = Cider().compute_score(self.ref_sentence, self.hypo_sentence)
        return cider_score[0]

    @property
    def rouge(self) -> np.float64:
        rouge_score = Rouge().compute_score(self.ref_sentence, self.hypo_sentence)
        return rouge_score[0]

    @property
    def meteor(self) -> float:
        meteor_score = Meteor().compute_score(self.ref_sentence, self.hypo_sentence)
        return meteor_score[0]

    @property
    def all_metrics(self) -> Tuple[Union[float, np.float64, Tuple[float]]]:
        """Return all metrics"""
        return self.belu, self.cider, self.rouge, self.meteor


def setup_corpus(
    references: List[List[List[int]]],
    candidates: List[List[int]],
    rev_word_map: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    ref_sentence = []
    hypo_sentence = []

    for cnt, each_image in enumerate(references):
        # ground truths
        cur_ref_sentence = []
        for cap in each_image:
            sentence = [rev_word_map[ix] for ix in cap]
            cur_ref_sentence.append(' '.join(sentence))

        ref_sentence.append(cur_ref_sentence)

        # predictions
        sentence = [rev_word_map[ix] for ix in candidates[cnt]]
        hypo_sentence.append([' '.join(sentence)])

    return ref_sentence, hypo_sentence
