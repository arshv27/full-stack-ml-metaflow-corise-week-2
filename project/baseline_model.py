
class Baseline_Majority_Classifier(self):

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
        baseline_pred_proba = 1
        baseline_pred = self.majority_label

        return baseline_pred_proba, baseline_pred

    def score(self, labels, predictions_proba, predictions):
        self.accuracy_score = accuracy_score(labels, predictions)
        self.auroc_score = roc_auc_score(labels, predictions_proba)

        return self.accuracy_score, self.auroc_score
