from fer import FER

detector = FER(mtcnn=True)

def predict_emotion(img_np):
    result = detector.detect_emotions(img_np)
    if not result:
        return "Unknown"
    emotions = result[0]["emotions"]
    return max(emotions, key=emotions.get)