import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
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

        X, y = df.iloc[:, 0:4], df.iloc[:, -1]

        encoder = LabelEncoder()
        encoder.fit(y)
        Y = encoder.transform(y)
        Y = np_utils.to_categorical(Y)
        return X, Y


class NeuralConfiguration(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def model_creation(self):

        print("Creating Sequential Model.")
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        print("Adding Optimizer in the Model.")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Fitting Data in th Model.")
        model.fit(self.X, self.y, epochs=30, batch_size=5)

        print("Working on Model Evaluation.")
        _, accuracy = model.evaluate(self.X, self.y)
        print('Accuracy: %.2f' % (accuracy * 100))

        print("Predicting Value for the Test Data-set.")
        predicted_prob = model.predict(self.X)
        predicted_data = model.predict_classes(self.X)
        return predicted_prob, predicted_data


if __name__ == "__main__":

    file_path_ = r"C:\Users\m4singh\Documents\AnalysisNoteBook\DeepLearning\Categorical\iris.csv"

    file_obj = DataAnalysis(file_path_)
    X_, Y_ = file_obj.load_data()

    neural_obj = NeuralConfiguration(X_, Y_)
    predicted_prob_, predicted_data_ = neural_obj.model_creation()

    print(predicted_prob_[:4])
    print(predicted_data_[:4])
