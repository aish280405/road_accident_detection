from keras.models import model_from_json
import numpy as np
import os


class AccidentDetectionModel(object):

    class_nums = ['Accident', 'Non Accident']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the model, supporting both legacy and Keras 3 filenames
        possible_weight_files = [model_weights_file]
        if model_weights_file.endswith('.h5') and not model_weights_file.endswith('.weights.h5'):
            possible_weight_files.append(model_weights_file.replace('.h5', '.weights.h5'))

        last_error = None
        for weight_file in possible_weight_files:
            if not os.path.exists(weight_file):
                continue
            try:
                self.loaded_model.load_weights(weight_file)
                last_error = None
                break
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds