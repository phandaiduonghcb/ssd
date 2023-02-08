import cv2
import imutils
import numpy
import tensorflow as tf
from inference import predict

class FaceDetector:
  def __init__(self, pretrained_path):
    '''
    pretrained_path: pretrained folder
    '''
    self.pretrained_path = pretrained_path
    self.net =  tf.keras.models.load_model(pretrained_path)
    self.required_im_shape = (300,300)
    self.confidence = 0.75
    self.iou_threshold = 0.4
    self.output_width = 640
    self.output_height = 480

  def __detect_array(self, image):
    w = self.output_width
    h = self.output_height
    image = imutils.resize(image, width=640, height=480)
    tf_image = tf.convert_to_tensor(image)
    normalized_box = predict(self.net, tf.expand_dims(tf.image.resize(tf_image,self.required_im_shape),0), self.confidence, self.iou_threshold)
    a = normalized_box[:,:2]
    b = normalized_box[:,2:] * tf.expand_dims(tf.convert_to_tensor((w,h,w,h),dtype=tf.float32),0)
    rects = tf.concat((a,b),axis=1)[:,2:].numpy()
    # loop over the bounding boxes
    for (x1, y1, x2, y2) in rects:
      # draw the face bounding box on the image
      cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    return image

  def detect_image(self, image_path, save_path=None):
    '''
    Used to detect faces in an image.
    '''
    im = cv2.imread(image_path)
    im = self.__detect_array(im)
    print("[INFO] Press any key to exit...")
    # cv2.imshow(image_path, im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if save_path is not None:
      cv2.imwrite(save_path, im)
    return im

  def detect_video(self, video_path, save_path=None):
    '''
    Used to detect faces in a video or from webcam
    video_path: 0 for webcam or a string for a video path.
    '''
    capture = cv2.VideoCapture(video_path)
 
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    if save_path is not None:
      videoWriter = cv2.VideoWriter(save_path, fourcc, 15.0 , (640, 480))
    
    while (True):
    
      ret, frame = capture.read()
      
      if ret:
        new_frame = self.__detect_array(frame)
        if save_path is not None:
          videoWriter.write(new_frame)
        # cv2.imshow(str(video_path), new_frame)
      else:
        break
  
      # if cv2.waitKey(1) == 27:
      #     break
    
    capture.release()
    if save_path is not None:
      videoWriter.release()
    
    # cv2.destroyAllWindows()


if __name__ == '__main__':
  detector = FaceDetector("/dl/ssd/trained_models/ssd_vgg16/best_model")
  # detector.detect_image('jisoo.jpg','jisoo_pred.jpg')
  detector.detect_video('Video Of People Walking.mp4',save_path='test.mp4')
  # detector.detect_video(0)