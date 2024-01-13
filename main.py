import config
from model import Yolo_v3
from utils import load_weights, pred_to_box, draw_boxes_on_img, detect_image

import tensorflow as tf
import os




img_path = 'images/football.jpg'

imgs = os.listdir('images/')

build_net = Yolo_v3(n_classes=config.NUM_CLASSES, model_size=config.MODEL_SIZE)
model = build_net.return_model()

load_weights(model, config.WEIGTHS_FILE, build_net.get_layers_idx())


detect_image(img_path, model)


# for i in imgs:
#     detect_image("images/" + i, model)







# win_name = 'Yolov3 detection'
# cv2.namedWindow(win_name)
# cap = cv2.VideoCapture('videos/football_match.mp4')
# frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
#               cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# try:
#     while True:
#         start = time.time()
#         ret, frame = cap.read()
#         if not ret:
#             break
#         resized_frame = tf.expand_dims(frame, 0)
#         resized_frame = tf.image.resize(resized_frame, config.MODEL_SIZE[:2])
#         pred = model.predict(resized_frame)
#         boxes, scores, classes, nums = pred_to_box(
#             pred, config.MODEL_SIZE,
#             max_output_size=config.MAX_OUTPUT_SIZE,
#             max_output_class=config.MAX_OUTPUT_CLASS,
#             iou_threshold=config.IOU_THRESHOLD,
#             confidence_threshold=config.CONFIDENCE_THRESHOLD
#         )
#         img = draw_boxes_on_img(frame, boxes, scores, classes, nums, config.CLASS_NAMES)
#         cv2.imshow(win_name, img)
#         stop = time.time()
#         seconds = stop - start
#         fps = 1 / seconds
#         print("Estimated frames per second : {0}".format(fps))
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
# finally:
#     cv2.destroyAllWindows()
#     cap.release()
#     print('Detections have been performed successfully.')
