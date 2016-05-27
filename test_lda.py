import numpy as np
import lda
import lda.datasets
from GSDMM import GSDMM

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
X.shape
X.sum()
model = GSDMM(n_topics=20, n_iter=10, random_state=910820)
model.fit(X) 
topic_word = model.topic_word_  
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    #topic_dist = topic_dist.toarray()[0]
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
