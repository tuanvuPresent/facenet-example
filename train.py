import os
import pickle

from recognizer_facenet import FaceRecognizerFaceNet


def train(folder):
    data = {}
    embedder = FaceRecognizerFaceNet()
    for label in os.listdir(folder):
        data[str(label)] = []
        for filename in os.listdir(os.path.join(folder, label)):
            detections = embedder.extract(os.path.join(os.path.join(folder, label), filename))
            if len(detections) > 0:
                e1 = detections[0].get('embedding')
                data[str(label)].append(e1)
    with open('data', 'wb') as file:
        pickle.dump(data, file)
    print('Train DONE...')


if __name__ == '__main__':
    train('train_image')
