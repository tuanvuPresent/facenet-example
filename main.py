import pickle

from recognizer_facenet import FaceRecognizerFaceNet

if __name__ == '__main__':
    embedder = FaceRecognizerFaceNet()
    with open('data.plk', 'rb') as file:
        data = pickle.load(file)

    print(embedder.predict_image('test_image/test.jpeg', data))
    faces1 = embedder.extract('test_image/test.jpeg')
    for face in faces1:
        print(embedder.predict_face(face, data))
