from itertools import combinations
from math import log
import tqdm

class coherence_score():
    def __init__(self, filepath):
        self.dic = {}

        for line in tqdm.tqdm(open(filepath, encoding="utf-8")):
            words = list(set(line.strip().split()))
            # wl
            for w in words:
                if w in self.dic:
                    self.dic[w] += 1
                else:
                    self.dic[w] = 1
            
            # wn
            com_list = list(combinations(words, 2))
            for c in com_list:
                if c[0] < c[1]:
                    ws = c[0]
                    wb = c[1]
                else:
                    ws = c[1]
                    wb = c[0]

                key = ws + "_" + wb
                if key in self.dic:
                    self.dic[key] += 1
                else:
                    self.dic[key] = 1


    def get_D1_D2(self, wl, wn):
        d1 = d2 = 0
        if wl in self.dic:
            d1 = self.dic[wl]
        
        if wl < wn:
            ws = wl
            wb = wn
        else:
            ws = wn
            wb = wl
        if ws + "_" + wb in self.dic:
            d2 = self.dic[ws+"_"+wb]
        
        return d1, d2  
      

    # topn <= 50
    def get_co_score(self, words, topn):
        words = words[:topn]
        N = len(words)
        score = 0
        for n in range(2, N+1):
            for l in range(1, n):
                d1, d2 = self.get_D1_D2(words[l-1], words[n-1])
                score += log((d2+1)/d1)
        return score

