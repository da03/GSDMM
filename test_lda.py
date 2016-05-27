import numpy as np
import lda
import lda.datasets
from GSDMM import GSDMM

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
X.shape
X.sum()
#model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model = GSDMM(n_topics=20, n_iter=100, random_state=910820)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    #topic_dist = topic_dist.toarray()[0]
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
