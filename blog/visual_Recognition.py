from watson_developer_cloud import VisualRecognitionV3
import json


visual_recognition = VisualRecognitionV3(
        '2018-03-19',
        iam_apikey='imJM4atcDPtX7OZlyZLx8tTf7nnAoctaFxid5lNotOfQ'
    )

img_path='./unnamed.jpg'
with open(img_path, 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
            threshold='0.6',
    classifier_ids='default').get_result()

print(json.dumps(classes, indent=2))
