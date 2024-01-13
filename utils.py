import numpy as np
import cv2
import tensorflow as tf

import config


def load_weights(model, weight_file, layers_idx):
    fp = open(weight_file, 'rb')
    np.fromfile(fp, dtype=np.int32, count=5)

    not_bn_conv = [layers_idx[-1], layers_idx[-1]-12, layers_idx[-1]-24]

    for nl in layers_idx:
        conv_layer = model.get_layer('conv_' + str(nl))

        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        # print("layer {}:  {}".format(nl, conv_layer))

        if nl not in not_bn_conv:
            norm_layer = model.get_layer('bnorm_' + str(nl))
            # print("layer {}:  {}".format(nl, norm_layer))
            size = np.prod(norm_layer.get_weights()[0].shape)

            bn_weights = np.fromfile(fp, dtype=np.float32, count=4*filters)
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
        else:
            conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(fp, dtype=np.float32, count=np.product(conv_shape))
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if nl not in not_bn_conv:
            norm_layer.set_weights(bn_weights)
            conv_layer.set_weights([conv_weights])
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()


def nms(inputs, model_size, max_output_size, max_output_class, iou_threshold, confidence_threshold):
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox = bbox/model_size[0]

    scores = confs*class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )


    return boxes, scores, classes, valid_detections


def pred_to_box(inputs, model_size, max_output_size, max_output_class, iou_threshold, confidence_threshold):
    x, y, w, h, conf, cl = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    topl_x = x-(w/2.0)
    topl_y = y-(h/2.0)
    botr_x = x+(w/2.0)
    botr_y = y+(h/2.0)

    inputs = tf.concat([topl_x, topl_y, botr_x, botr_y, conf, cl], axis=-1)

    output_boxes = nms(inputs, model_size, max_output_size, max_output_class, iou_threshold, confidence_threshold)

    return output_boxes


def draw_boxes_on_img(img, boxes, ious, classes, nums, class_names):
    boxes, ious, classes, nums = boxes[0], ious[0], classes[0], nums[0]
    boxes=np.array(boxes)
    for i in range(nums):
        color = config.CLASS_COLORS[int(classes[i])]

        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))
        img = cv2.rectangle(img, (x1y1), (x2y2), color, 1)


        dx = int(abs(x2y2[0] - x1y1[0]) / 10)
        dy = int(abs(x2y2[1] - x1y1[1]) / 10)
        #1 corner
        img = cv2.line(img, x1y1, (x1y1[0] + dx, x1y1[1]), color, 2)
        img = cv2.line(img, x1y1, (x1y1[0], x1y1[1] + dy), color, 2)
        # 2 corner
        img = cv2.line(img, (x2y2[0], x1y1[1]), (x2y2[0] - dx, x1y1[1]), color, 2)
        img = cv2.line(img, (x2y2[0], x1y1[1]), (x2y2[0], x1y1[1] + dy), color, 2)
        # 1 corner
        img = cv2.line(img, (x1y1[0], x2y2[1]), (x1y1[0] + dx, x2y2[1]), color, 2)
        img = cv2.line(img, (x1y1[0], x2y2[1]), (x1y1[0], x2y2[1] - dy), color, 2)
        # 1 corner
        img = cv2.line(img, x2y2, (x2y2[0] - dx, x2y2[1]), color, 2)
        img = cv2.line(img, x2y2, (x2y2[0], x2y2[1] - dy), color, 2)

        iou = int(ious[i] * 100)/100.0
        text = '{} {}'.format(class_names[int(classes[i])], iou)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
        img = cv2.rectangle(img,
                            (x1y1[0] - 1, x1y1[1] - text_size[1] - baseline),
                            (x1y1[0] + text_size[0], x1y1[1]),
                            color,
                            thickness=cv2.FILLED)

        img = cv2.putText(img,
                          text,
                          (x1y1[0], x1y1[1] - baseline),
                          cv2.FONT_HERSHEY_PLAIN,
                          fontScale=0.5,
                          color=(255, 255, 255),
                          thickness=1)
    return img


def detect_image(img_path, model):
    image = cv2.imread(img_path)
    image = np.array(image)
    image = tf.expand_dims(image, 0)
    resized_img = tf.image.resize(image, config.MODEL_SIZE[:2])
    pred = model.predict(resized_img)

    boxes, ious, classes, nums = pred_to_box(pred, config.MODEL_SIZE,
                                            max_output_size=config.MAX_OUTPUT_SIZE,
                                            max_output_class=config.MAX_OUTPUT_CLASS,
                                            iou_threshold=config.IOU_THRESHOLD,
                                            confidence_threshold=config.CONFIDENCE_THRESHOLD)


    image = np.squeeze(image)
    img = draw_boxes_on_img(image, boxes, ious, classes, nums, config.CLASS_NAMES)

    cv2.imshow("predict", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
