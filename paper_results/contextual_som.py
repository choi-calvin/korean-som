from konlpy.tag import Mecab
import numpy as np
from minisom import MiniSom
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


class ContextualSom:
    def __init__(self, corpus):
        self._corpus = corpus
        self._tokens = []
        self._token_to_vector = {}
        self._token_to_avg_vector = {}
        self._som = None
        self._all_labels = ["noun", "verb", "closed_class",
          "quantifier", "classifier", "adjective",
          "adverb", "interjection", "unknown"]

    def _average_vector(self, token):
        before = np.zeros(100)
        after = np.zeros(100)
        before_count = 0
        after_count = 0

        # Sweep a window through processed corpus
        # Calculate the average of all the vectors appearing before and after the token
        for i in range(len(self._tokens)):
            if self._tokens[i] == token:
                if i > 0 and self._tokens[i - 1] in self._token_to_vector:
                    before += self._token_to_vector[self._tokens[i - 1]]
                    before_count += 1
                if i < len(self._tokens) - 2 and self._tokens[i + 1] in self._token_to_vector:
                    after += self._token_to_vector[self._tokens[i + 1]]
                    after_count += 1

        if before_count != 0:
            before = before / before_count
        if after_count != 0:
            after = after / after_count

        return normalize(np.concatenate([before, after]))

    @staticmethod
    def _get_category(pos):
        if pos in ["NNG", "NNP"]:
            return "noun"
        if pos == "VV":
            return "verb"
        if pos in ["VA"]:
            return "adjective"
        if pos in ["NR", "SN"]:
            return "quantifier"
        if pos == "NNBC":
            return "classifier"
        if pos == "MAG":
            return "adverb"
        if pos == "IC":
            return "interjection"
        if pos == "UNKNOWN":
            return "unknown"

        return "closed_class"

    @staticmethod
    def _get_colour(category):
        map_ = {
            "noun": "yellow",
            "verb": "blue",
            "closed_class": "red",
            "quantifier": "pink",
            "classifier": "cyan",
            "adjective": "green",
            "adverb": "orange",
            "interjection": "purple",
            "unknown": "gray"
        }

        return map_[category]

    def preprocess(self):
        mecab = Mecab()  # Parts-of-speech tagger
        token_pos = mecab.pos(self._corpus)

        # Mecab sometimes returns multiple POS tags for a token; we take the first one for simplicity
        self._tokens = [(token, pos.split("+")[0]) for token, pos in token_pos]

        counter = Counter(self._tokens)
        counter = { token: count for token, count in counter.most_common(500) }

        # Assign random vectors to each token
        self._token_to_vector = { token: normalize(np.random.normal(size=100)) for token in counter }
        self._token_to_avg_vector = { token: self._average_vector(token) for token in counter }
    
    def train(self, x, y, epochs, verbose=False, **kwargs):
        som_input = np.asarray(list(self._token_to_avg_vector.values()))

        # All hyperparameters from Zhao, Li, et al., 2011
        self._som = MiniSom(x, y, som_input.shape[1], **kwargs)
        self._som.train(som_input, epochs, verbose=verbose)
    
    def scores(self):
        positions = []
        labels = []

        for token, v in self._token_to_avg_vector.items():
            labels.append(self._get_category(token[1]))
            positions.append(self._som.winner(v))

        positions = np.asarray(positions)
        labels = np.asarray(labels)
        label_ind = np.asarray([self._all_labels.index(l) for l in labels])

        predictions = []
        for ind, p in enumerate(positions):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(np.delete(positions, ind, axis=0), np.delete(label_ind, ind, axis=0))
            predictions.append(knn.predict([p])[0])

        predictions = np.asarray(predictions)
        
        scores = {}

        for label in range(len(self._all_labels)):
            cat_labels = label_ind[label_ind == label]
            cat_predictions = predictions[label_ind == label]

            correct = cat_labels == cat_predictions
            correct_percentage = correct.sum() / len(cat_labels)
            
            scores[self._all_labels[label]] = correct_percentage
        
        return scores