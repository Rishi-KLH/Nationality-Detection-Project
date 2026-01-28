from deepface import DeepFace

def predict_age(img_np):
    result = DeepFace.analyze(img_np, actions=["age"], enforce_detection=False)
    return result[0]["age"]