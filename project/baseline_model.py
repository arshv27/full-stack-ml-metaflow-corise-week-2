
from sklearn.metrics import roc_auc_score, accuracy_score

class Baseline_Majority_Classifier():

    def __init__(self, train_df, val_df):
        self.train_data = train_df
        self.val_data = val_df

    def get_majority_label(self, train_y):
        frequency_counts = train_y.value_counts()
        majority_label = frequency_counts.idxmax()

        return majority_label

    def fit(self, train_X, train_y):
        self.majority_label = self.get_majority_label(train_y)

        return

    def predict(self, infer_X):
        baseline_pred_proba = [1]*(len(infer_X))
        baseline_pred = [self.majority_label]*(len(infer_X))

        return baseline_pred_proba, baseline_pred

    def score(self, labels, predictions_proba, predictions):
        
        self.accuracy = accuracy_score(labels, predictions)
        self.auroc = roc_auc_score(labels, predictions_proba)

        return self.accuracy, self.auroc
