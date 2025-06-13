# Copyright (C) CVAT AI Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolov8l.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False

        if cuda:
            try:
                # Check if CUDA is available
                ort.get_device()
            except Exception as e:
                print(f"CUDA is not properly configured. Falling back to CPU: {e}")
                cuda = False

        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=session_options)
            self.output_names = [i.name for i in self.model.get_outputs()]
            self.input_names = [i.name for i in self.model.get_inputs()]

            self.is_initiated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_shape_unpadded = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_shape_unpadded[0], new_shape[0] - new_shape_unpadded[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_shape_unpadded:  # resize
            im = cv2.resize(im, new_shape_unpadded, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _nms(self, boxes, scores, threshold=0.5, iou_threshold=0.5):
        # boxes: (N, 4), scores: (N,)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=threshold,
            nms_threshold=iou_threshold
        )
        if len(indices) > 0:
            indices = indices.flatten()
            return indices
        else:
            return []

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, pad = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32)
            im /= 255

            inp = {self.input_names[0]: im}
            detections = self.model.run(self.output_names, inp)[0]

            detections = detections.transpose((0, 2, 1))  # (1, 8400, 84)

            boxes = detections[..., :4]  # (1, 8400, 4)
            scores = np.max(detections[..., 4:], axis=-1).squeeze()  # (8400,)
            labels = np.argmax(detections[..., 4:], axis=-1).squeeze()  # (8400,)

            boxes = boxes.reshape(-1, 4)

            # Convert xywh to xyxy
            boxes_xyxy = boxes.copy()
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            boxes = boxes_xyxy

            pad = np.array(pad * 2)
            boxes -= pad
            boxes /= ratio
            boxes = boxes.round().astype(np.int32)

            # Apply NMS here
            nms_indices = self._nms(boxes, scores, threshold=0.5, iou_threshold=0.5)
            boxes = boxes[nms_indices]
            labels = labels[nms_indices]
            scores = scores[nms_indices]

            return [boxes, labels, scores]

        except Exception as e:
            print(e)

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            for label, score, box in zip(labels, scores, boxes):
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)

                    results.append({
                        "confidence": str(score),
                        "label": self.labels.get(label, "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    })

        return results
