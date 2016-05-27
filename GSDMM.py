import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import find
import math

class GSDMM:
    def __init__(self, n_topics, n_iter, random_state=910820, alpha=0.1, beta=0.1):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.random_state = random_state
        np.random.seed(random_state)
        self.alpha = alpha
        self.beta = beta
    def fit(self, X):
        alpha = self.alpha
        beta = self.beta

        D, V = X.shape
        K = self.n_topics

        N_d = X.sum(axis=1)
        words_d = {}
        for d in range(D):
            words_d[d] = find(X[d,:])[1]

        # initialization
        N_k = np.zeros(K)
        M_k = np.zeros(K)
        N_k_w = lil_matrix((K, V), dtype=np.int32)

        K_d = np.zeros(D)

        for d in range(D):
            k = np.random.choice(K, 1, p=[1.0/K]*K)[0]
            K_d[d] = k
            M_k[k] = M_k[k]+1
            N_k[k] = N_k[k] + N_d[d]
            for w in words_d[d]:
                N_k_w[k, w] = N_k_w[k,w]+X[d,w]

        for iter in range(self.n_iter):
            print 'iter ', iter
            for d in range(D):
                k_old = K_d[d]
                M_k[k_old] -= 1
                N_k[k_old] -= N_d[d]
                for w in words_d[d]:
                    N_k_w[k_old, w] -= X[d,w]
                # sample k_new
                log_probs = [0]*K
                for k in range(K):
                    log_probs[k] += math.log(alpha+M_k[k])
                    for w in words_d[d]:
                        N_d_w = X[d,w]
                        for j in range(N_d_w):
                            log_probs[k] += math.log(N_k_w[k,w]+beta+j)
                    for i in range(N_d[d]):
                        log_probs[k] -= math.log(N_k[k]+beta*V+i)
                log_probs = np.array(log_probs) - max(log_probs)
                probs = np.exp(log_probs)
                probs = probs/np.sum(probs)
                k_new = np.random.choice(K, 1, p=probs)[0]
                K_d[d] = k_new
                M_k[k_new] += 1
                N_k[k_new] += N_d[d]
                for w in words_d[d]:
                    N_k_w[k_new, w] += X[d,w]
        self.topic_word_ = N_k_w.toarray()

            

