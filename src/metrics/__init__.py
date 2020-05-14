from .bleu import bleu
from .cider import cider
from .meteor import meteor
from .rouge import rouge

'''
class Metrics: compute metrics on given reference and candidate sentences set
               now supports BLEU, CIDEr, METEOR and ROUGE-L

input params:
    references([[[ref1a], [ref1b], [ref1c]], ..., [[refna], [refnb]]]): reference sencences (word id)
    candidates([[hyp1], [hyp2], ..., [hypn]]): candidate sencences (word id)
    rev_word_map: ix2word map
''' 
class Metrics:

    def __init__(self, references, candidates, rev_word_map):
        corpus = setup_corpus(references, candidates, rev_word_map)
        self.ref_sentence = corpus[0] 
        self.hypo_sentence = corpus[1] 

    '''
    compute BLEU scores
    return:
        BLEU-1(float), BLEU-2(float), BLEU-3(float), BLEU-4(float)
    ''' 
    def belu(self):
        bleu_score = bleu.Bleu().compute_score(self.ref_sentence, self.hypo_sentence)
        return bleu_score[0][0], bleu_score[0][0], bleu_score[0][2], bleu_score[0][3]

    '''
    compute CIDEr scores
    return: CIDEr(float)
    ''' 
    def cider(self):
        cider_score = cider.Cider().compute_score(self.ref_sentence, self.hypo_sentence)
        return cider_score[0]

    '''
    compute CIDEr scores
    return: ROUGE-L(float)
    ''' 
    def rouge(self):
        rouge_score = rouge.Rouge().compute_score(self.ref_sentence, self.hypo_sentence)
        return rouge_score[0]

    '''
    compute CIDEr scores
    return: METEOR(float)
    '''
    def meteor(self):
        meteor_score = meteor.Meteor().compute_score(self.ref_sentence, self.hypo_sentence)
        return meteor_score[0]

    '''
    compute all meterics
    return:
        BLEU-1(float), BLEU-2(float), BLEU-3(float), BLEU-4(float),
        CIDEr(float), ROUGE-L(float), METEOR(float)
    ''' 
    def all_metrics(self):
        return self.belu(), self.cider(), self.rouge(), self.meteor()


def setup_corpus(references, candidates, rev_word_map):

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