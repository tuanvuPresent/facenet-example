from keras_facenet import FaceNet


class FaceRecognizerFaceNet(FaceNet):
    def __init__(
            self,
            key='20180402-114759',
            use_prebuilt=True,
    ):
        super(FaceRecognizerFaceNet, self).__init__(key, use_prebuilt, cache_folder='facenet_model/')

    def compare(self, filepath_or_image1, filepath_or_image2):
        try:
            detections = self.extract(filepath_or_image1)
            detections2 = self.extract(filepath_or_image2)
            if len(detections) > 0 and len(detections2) > 0:
                e1 = detections[0].get('embedding')
                e2 = detections2[0].get('embedding')
                # return 1 - np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                return self.compute_distance(e1, e2)
        except:
            raise Exception('Images not include face or url not exists')

    def predict_image(self, filepath_or_image, data=None, distance=0.35):
        detections = self.extract(filepath_or_image)
        if len(detections) == 0:
            raise Exception('Image not include face')
        if data is None:
            return None, 1

        e1 = detections[0].get('embedding')
        index = {}
        for label, value in data.items():
            distance_list = []
            for e2 in value:
                distance_list.append(self.compute_distance(e1, e2))
            index[label] = min(distance_list)

        best_label = min(index, key=index.get)
        best_distance = index[best_label]
        if best_distance < distance:
            return best_label, best_distance
        else:
            return None, best_distance

    def predict_face(self, face, data=None, distance=0.35):
        if data is None:
            return None, 1

        e1 = face.get('embedding')
        index = {}
        for label, value in data.items():
            distance_list = []
            for e2 in value:
                distance_list.append(self.compute_distance(e1, e2))
            index[label] = min(distance_list)

        best_label = min(index, key=index.get)
        best_distance = index[best_label]
        if best_distance < distance:
            return best_label, best_distance
        else:
            return None, best_distance
