import numpy as np
import pdb

'''
calculates longest common subsequence for a pair of tokenized strings

input params:
    string(list of str): tokens from a string split using whitespace
    sub(list of str): shorter string, also split using whitespace

returns: 
    length(list of int): length of the longest common subsequence between the two strings

Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
'''
def my_lcs(string, sub):

    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

'''
class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
'''
class Rouge():

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    '''
    compute ROUGE-L score given one candidate and references for an image

    input params:
        candidate(str): candidate sentence to be evaluated
        refs(list of str): reference sentences for the particular image to be evaluated
    return:
        score(int): ROUGE-L score for the candidate evaluated against references
    '''
    def calc_score(self, candidate, refs):

        # assert(len(candidate)==0)
        # assert(len(refs)>0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs/float(len(token_c)))
            rec.append(lcs/float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if(prec_max!=0 and rec_max !=0):
            score = ((1 + self.beta**2)*prec_max*rec_max)/float(rec_max + self.beta**2*prec_max)
        else:
            score = 0.0
        return score


    '''
    compute Rouge-L score given a set of reference and candidate sentences for the dataset

    input params:
        reference(list): reference sentences ([[ref1a, ref1b, ref1c], ..., [refna, refnb]])
        hypothesis(list): predicted sentences ([[hypo1], [hypo2], ..., [hypon]])
    
    return: 
        average_score(float): mean ROUGE-L score computed by averaging scores for all the images
        scores: ROUGE-L scores computed for each image
    '''
    def compute_score(self, reference, hypothesis):
        
        assert len(reference) == len(hypothesis)

        score = []
        for id, hypo in enumerate(hypothesis):

            hypo = hypo
            ref = reference[id]

            # sanity check
            assert(type(hypo) is list)
            assert(len(hypo) >= 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)

            score.append(self.calc_score(hypo, ref))

        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"