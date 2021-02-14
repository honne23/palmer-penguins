import typing
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV


class PenguinSVC:

    def __init__(self, df: pd.DataFrame, test_size: float, features: typing.List[str], include_sex=False):
        svc = SVC(kernel='rbf')
        distributions = dict(kernel=['linear', 'rbf'], C=uniform(
            0.1, 1000), gamma=uniform(0.1, 5))
        self.clf = RandomizedSearchCV(svc, distributions, random_state=0)
        self.training_set, self.test_set = self.create_train_test_split(
            df, test_size, features, include_sex=include_sex)

    def create_train_test_split(self, df: pd.DataFrame, test_size: float, features: typing.List[str], include_sex=False):
        """
        Create training and test splits to train classifer models, optionally include sex as a feature
        """
        df_slice = df[['species', *features]]
        if include_sex:
            X = df.drop('species', axis=1)
        else:
            X = df.drop(['species', 'sex'], axis=1)
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)
        return (X_train, y_train), (X_test, y_test)

    def train(self):
        """
        Traing the support vector machine
        """
        start = time.time()
        self.clf.fit(self.training_set[0], self.training_set[1])
        end = time.time()

        print(f'Elapsed time in seconds: {round(end - start, 5)}s\n')

        y_pred = clf.predict(self.test_set[0])

        print(classification_report(self.test_set[1], y_pred))
