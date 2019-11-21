import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

# Create a label (category) encoder object
le = LabelEncoder()


class DataAnalysis(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)

        print("Data Description :: ")
        print(df.describe())
        return df.iloc[:, 0:8], df.iloc[:, -1]


class NeuralConfiguration(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model_creation(self):

        print("Creating Sequential Model.")
        model = Sequential()
        model.add(Dense(16, input_dim=8, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1, activation="softmax"))

        print("Adding Optimizer in the Model.")
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        print("Fitting Data in th Model.")
        model.fit(self.X, self.y, epochs=150, batch_size=40)

        print("Working on Model Evaluation.")
        _, accuracy = model.evaluate(self.X, self.y)
        print('Accuracy: %.2f' % (accuracy * 100))

        print("Predicting Value for the Test Data-set.")
        predicted_prob = model.predict(self.X)
        predicted_data = model.predict_classes(self.X)
        return predicted_prob, predicted_data


if __name__ == "__main__":

    file_path_ = r"C:\Users\m4singh\Documents\AnalysisNoteBook\DeepLearning\Categorical\diabetes_data.csv"

    file_obj = DataAnalysis(file_path_)
    X_, y_ = file_obj.load_data()

    neural_obj = NeuralConfiguration(X_, y_)
    predicted_prob_, predicted_data_ = neural_obj.model_creation()

    print(predicted_prob_[:4])
    print(predicted_data_[:4])
