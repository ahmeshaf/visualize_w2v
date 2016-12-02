import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import gensim

from nltk.corpus import stopwords
# cachedStopWords = stopwords.words("english")

# FILE_NAME = './models/dum_mod.mdl'

def main():
    embeddings_file = sys.argv[1]
    model = gensim.models.Word2Vec.load(embeddings_file)
    inpt_word = str(sys.argv[2])
    top_n = int(sys.argv[3])
    top_n_closest = [word[0] for word in model.most_similar(inpt_word, topn = top_n)]
    
    vec_array = [model[word] for word in top_n_closest]

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vec_array)

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(top_n_closest, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'usage: visualize.py mdl_path word top_n_closest'
        exit()
    main()
