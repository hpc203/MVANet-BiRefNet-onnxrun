import cv2
import onnxruntime
import numpy as np


class MVANet:
    def __init__(self, modelpath):
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # Initialize model
        # net = cv2.dnn.readNet(modelpath)  ###读取失败
        self.onnx_session = onnxruntime.InferenceSession(modelpath, so)
        self.input_name = self.onnx_session.get_inputs()[0].name

        input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))

    def prepare_input(self, image):
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(
            self.input_width, self.input_height))
        input_image = (input_image.astype(np.float32) / 255.0 - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def detect(self, image, score_th=None):
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.onnx_session.run(None, {self.input_name: input_image})

        # Post process: Squeeze, Sigmoid, Multiply by 255, uint8 cast
        mask = np.squeeze(result[-1])
        mask = 1 / (1 + np.exp(-mask))
        if score_th is not None:
            mask = np.where(mask < score_th, 0, 1)
        mask *= 255
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        return mask
    

if __name__ == '__main__':
    imgpath = 'testimgs/4.jpg'

    mynet = MVANet('mvanet_1024x1024.onnx')
    srcimg = cv2.imread(imgpath)
    mask = mynet.detect(srcimg, score_th=0.9)

    temp_image = np.zeros_like(srcimg) + 255
    mask = np.stack((mask, ) * 3, axis=-1).astype(np.uint8)
    dstimg = np.where(mask, srcimg, temp_image)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', mask)
    cv2.namedWindow('dstimg', cv2.WINDOW_NORMAL)
    cv2.imshow('dstimg', dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()