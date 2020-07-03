import pathlib

import numpy as np
import tensorflow as tf
import cv2

dataset_path = pathlib.Path('/moby/datasets/deepercut/volumental')
cropped_path = dataset_path / 'cropped'
cropped_keypoints_path = cropped_path / 'keypoints'

def detection_path_from_image_path(image_path):
    return detections_dir / f'{image_path.stem}.json'


def annotation_path_from_image_path(image_path):
    json_stem = image_path.stem.replace('__color', '')
    return annotations_path / (json_stem + '.json')


def keypoint_path_from_cropped_image_path(image_path):
    return cropped_keypoints_path / f'{image_path.stem}.json'


def ious(first_bboxes, second_bboxes):
    """Computes IoUs between two sets of bounding boxes.
    args:
        first_bboxes: List of bounding boxes in COCO format (x_min, y_min, width, height)
        second_bboxes: List of bounding boxes in COCO format (x_min, y_min, width, height)
    returns:
        numpy array of size len(first_bboxes) x len(second_bboxes) with IOUs
    """
    return cocoUtils.iou(first_bboxes, second_bboxes, [0])


def select_best_detection(detected_bboxes, gt_bbox):
    if len(detected_bboxes) < 1:
        return None
    
    detection_ious = ious(detected_bboxes, [gt_bbox])
    best_index = np.argmax(detection_ious)
    best_iou = detection_ious[best_index]
    if best_iou > 0.2:
        return detected_bboxes[best_index]
    return None


_keypoint_to_id = {'arch_point': 0,
                   'heel': 1,
                   'instep': 2,
                   'toe': 3,
                   'width_lateral': 4,
                   'width_medial': 5}
def keypoint_to_id(keypoint):
    return keypoint_to_id[keypoiint]


class TFLiteKeyPointRegressor:
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.image_input_index = input_details[0]['index']
        self.mu_output_index = output_details[1]['index']
        self.sigma_logits_output_index = output_details[2]['index']
        self.is_right_output_index = output_details[0]['index']
        
        self.interpreter.allocate_tensors()
        
    def __call__(self, image):
        self.interpreter.set_tensor(self.image_input_index, image)
        self.interpreter.invoke()
    
        is_right = self.interpreter.get_tensor(self.is_right_output_index)
        mu = self.interpreter.get_tensor(self.mu_output_index)
        sigma_logits = self.interpreter.get_tensor(self.sigma_logits_output_index)

        mu = np.where(
                is_right < 0.5,
                mu[:, : 2 * 6],
                mu[:, 2 * 6:]
            )
        sigma_logits = np.where(
                is_right < 0.5,
                sigma_logits[:, : 2 * 6],
                sigma_logits[:, 2 * 6:]
            )
        
        mu = mu.reshape((-1, 2))
        sigma_logits = sigma_logits.reshape((-1, 2))

        return mu, sigma_logits, is_right
    

def predict_with_regressor(image, key_point_regressor):
    target_size = np.max(image.shape[:2])
    diff = (target_size - image.shape[:2]) / 2
    padding = [2*[int(w)] for w in list(diff)+ [0]]
    padded_image = np.pad(image, padding, constant_values=0.0)
    resized_image = cv2.resize(padded_image, dsize=(224, 224))
    
    resized_image = np.float32(resized_image / 225)[np.newaxis, ...]
        
    mu, sigma, is_right = key_point_regressor(resized_image)
    
    mu = mu @ np.diag(padded_image.shape[:2])
    
    mu = mu - diff[::-1]
    return mu