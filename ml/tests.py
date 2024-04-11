import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
        self.y_train = [0, 1, 0, 1]
        self.model.fit(self.X_train, self.y_train)

    def test_accuracy(self):
        # Test the accuracy of the model
        X_test = [[0, 0], [1, 1], [2, 2], [3, 3]]
        y_test = [0, 1, 0, 1]
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertAlmostEqual(accuracy, 1.0)  # Assert that accuracy is close to 100%

    def test_precision(self):
        # Test the precision of the model
        X_test = [[0, 0], [1, 1], [2, 2], [3, 3]]
        y_test = [0, 1, 0, 1]
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        self.assertAlmostEqual(precision, 1.0)  # Assert that precision is close to 100%

    def test_recall(self):
        # Test the recall of the model
        X_test = [[0, 0], [1, 1], [2, 2], [3, 3]]
        y_test = [0, 1, 0, 1]
        y_pred = self.model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        self.assertAlmostEqual(recall, 1.0)  # Assert that recall is close to 100%

    def test_f1_score(self):
        # Test the F1 score of the model
        X_test = [[0, 0], [1, 1], [2, 2], [3, 3]]
        y_test = [0, 1, 0, 1]
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        self.assertAlmostEqual(f1, 1.0)  # Assert that F1 score is close to 100%

    def test_prediction(self):
        # Test individual predictions
        X_test = [[0, 0], [1, 1], [2, 2], [3, 3]]
        y_pred = self.model.predict(X_test)
        self.assertListEqual(list(y_pred), [0, 1, 0, 1])  # Assert predicted labels


if __name__ == "__main__":
    unittest.main()
