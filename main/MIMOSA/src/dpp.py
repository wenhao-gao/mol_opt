import numpy as np
import math
np.random.seed(1)

class DPPModel(object):
    def __init__(self, smiles_lst, sim_matrix, f_scores, top_k, lamb):
        self.smiles_lst = smiles_lst 
        self.sim_matrix = sim_matrix # (n,n)
        self.lamb = lamb
        self.f_scores = np.exp(f_scores) * self.lamb # (n,) 
        self.max_iter = top_k 
        self.n = len(smiles_lst)
        self.kernel_matrix = self.f_scores.reshape((self.n, 1)) \
                             * sim_matrix * self.f_scores.reshape((1, self.n))
        self.log_det_V = np.sum(f_scores) * self.lamb 
        self.log_det_S = np.log(np.linalg.det(np.mat(self.kernel_matrix)))

    def dpp(self): 
        c = np.zeros((self.max_iter, self.n))
        d = np.copy(np.diag(self.kernel_matrix))  ### diagonal
        j = np.argmax(d)
        Yg = [j]
        _iter = 0
        Z = list(range(self.n))
        while len(Yg) < self.max_iter:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if _iter == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:_iter, j], c[:_iter, i])) / np.sqrt(d[j])
                c[_iter, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            Yg.append(j)
            _iter += 1

        return [self.smiles_lst[i] for i in Yg], self.log_det_V, self.log_det_S 



if __name__ == "__main__":
    rank_score = np.random.random(size=(100)) 
    item_embedding = np.random.randn(100, 5) 
    item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
    sim_matrix = np.dot(item_embedding, item_embedding.T) 

    dpp = DPPModel(smiles_lst=list(range(100)), sim_matrix = sim_matrix, f_scores = rank_score, top_k = 10)
    Yg = dpp.dpp() 
    print(Yg)




