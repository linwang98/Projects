import numpy as np
import pickle
from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range
        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []
        # The evaluated classifier (see classifier.py)
        self.classifier = classifier
    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='latin1')
            #(self.train_embeddings, self.train_labels) = pickle.load(f)
            #(self.train_embeddings, self.train_labels) = cPickle.load(f, encoding='latin1')
        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='latin1')

    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):
        self.classifier.fit(self.train_embeddings, self.train_labels)
        #Predict similarities on the test data.
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)
        self.similarities = similarities
        similarity_threshold = self.select_similarity_threshold(similarities, self.false_alarm_rate_range*100)
        self.similarity_thresholds = similarity_threshold

        identification_rates = self.calc_identification_rate(prediction_labels)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_threshold,
                              'identification_rates': identification_rates}
        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        threshold=[]
        for i in false_alarm_rate:
            threshold = np.append(threshold,np.percentile(similarity,i))

        return threshold


    def calc_identification_rate(self, prediction_labels):
        rr = []
        for j in range(len(self.similarity_thresholds)):
            sum_test = 0.0
            num_knows = 0.0
            for i in range(len(prediction_labels)):
                if (self.similarities[i] >= self.similarity_thresholds[j]):
                    num_knows += 1
                    if (prediction_labels[i]==self.test_labels[i]):
                        sum_test+=1
            rate = sum_test / num_knows
            rr = np.append(rr,rate)
        return rr
