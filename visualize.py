import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import gensim

from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

# FILE_NAME = './models/dum_mod.mdl'

def main():
    embeddings_file = sys.argv[1]
    model = gensim.models.Word2Vec.load(embeddings_file)
    top_n = int(sys.argv[2])
    top_100_words = model.index2word[:top_n]
    wo_stopwords = [word for word in top_100_words if word not in cachedStopWords and len(word) > 1]

    vec_array = [model[word] for word in wo_stopwords]

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vec_array)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(wo_stopwords, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in
                               f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'usage: visualize.py mdl_path top_n_words'
        exit()
    main()
