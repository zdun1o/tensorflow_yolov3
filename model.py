import config
import tensorflow as tf


class Yolo_v3:
    def __init__(self, n_classes, model_size):

        self.n_classes = n_classes
        self.model_size = model_size
        self.i = 0
        self.conv_layers_idx = []


    def return_model(self):
        inputs = input_image = tf.keras.layers.Input(shape=self.model_size)
        inputs = inputs / 255.0

        route1, route2, inputs = self.darknet53(inputs)

        inputs, conv_sobj = self.yolo_conv_block(inputs, filters=512)

        detect1 = self.yolo_detect(conv_sobj, n_classes=self.n_classes, anchors=config.ANCHORS[6:9],
                              img_size=self.model_size)

        inputs = self.conv_block(inputs, 256, 1, 1)
        inputs = tf.keras.layers.UpSampling2D(2)(inputs)
        self.i+=1
        inputs = tf.concat([inputs, route2], axis=-1)
        self.i += 1

        inputs, conv_mobj = self.yolo_conv_block(inputs, filters=256)

        detect2 = self.yolo_detect(conv_mobj, n_classes=self.n_classes, anchors=config.ANCHORS[3:6],
                              img_size=self.model_size)

        inputs = self.conv_block(inputs, 128, 1, 1)
        inputs = tf.keras.layers.UpSampling2D(2)(inputs)
        self.i += 1
        inputs = tf.concat([inputs, route1], axis=-1)
        self.i += 1

        inputs, conv_lobj = self.yolo_conv_block(inputs, filters=128)

        detect3 = self.yolo_detect(conv_lobj, n_classes=self.n_classes, anchors=config.ANCHORS[0:3],
                              img_size=self.model_size)

        out_pred = tf.concat([detect1, detect2, detect3], axis=1)

        model = tf.keras.Model(input_image, out_pred)
        model.summary()

        return model

    def conv_block(self, inputs, filters, kernel_size, stride):
        inputs = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        use_bias=False,
                                        padding='valid' if stride > 1 else 'same',
                                        name="conv_" + str(self.i))(inputs)
        inputs = tf.keras.layers.BatchNormalization(name="bnorm_" + str(self.i))(inputs)
        inputs = tf.keras.layers.LeakyReLU(alpha=config.LEAKY_RELU)(inputs)
        self.conv_layers_idx.append(self.i)
        self.i+=1
        return inputs


    def residual_block(self, inputs, filters):
        shortcut = inputs
        conv = self.conv_block(inputs, filters, 1, 1)
        conv = self.conv_block(conv, 2 * filters, 3, 1)
        res_output = shortcut + conv
        self.i+=1
        return res_output


    def darknet53(self, inputs):
        inputs = self.conv_block(inputs, 32, 3, 1)

        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = self.conv_block(inputs, 64, 3, 2)

        for _ in range(1):
            inputs = self.residual_block(inputs, 32)

        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = self.conv_block(inputs, 128, 3, 2)

        for _ in range(2):
            inputs = self.residual_block(inputs, 64)

        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = self.conv_block(inputs, 256, 3, 2)

        for _ in range(8):
            inputs = self.residual_block(inputs, 128)

        route_1 = inputs

        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = self.conv_block(inputs, 512, 3, 2)

        for _ in range(8):
            inputs = self.residual_block(inputs, 256)

        route_2 = inputs

        inputs = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        inputs = self.conv_block(inputs, 1024, 3, 2)

        for _ in range(4):
            inputs = self.residual_block(inputs, 512)

        return route_1, route_2, inputs


    def yolo_conv_block(self, inputs, filters):
        inputs = self.conv_block(inputs, filters, 1, 1)
        inputs = self.conv_block(inputs, 2 * filters, 3, 1)
        inputs = self.conv_block(inputs, filters, 1, 1)
        inputs = self.conv_block(inputs, 2 * filters, 3, 1)
        inputs = self.conv_block(inputs, filters, 1, 1)
        conv_lobj = self.conv_block(inputs, 2 * filters, 3, 1)

        return inputs, conv_lobj


    def yolo_detect(self, inputs, n_classes, anchors, img_size):
        n_anchors = len(anchors)

        inputs = tf.keras.layers.Conv2D(filters=n_anchors * (5 + n_classes),
                                        kernel_size=1,
                                        strides=1,
                                        use_bias=True,
                                        padding='same',
                                        name="conv_" + str(self.i))(inputs)
        self.conv_layers_idx.append(self.i)
        self.i+=3

        shape = inputs.get_shape().as_list()
        grid_shape = shape[1:3]
        inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1], 5 + n_classes])

        box_centers, box_shapes, confidence, classes = tf.split(inputs, [2, 2, 1, n_classes], axis=-1)

        box_centers = tf.nn.sigmoid(box_centers)
        confidence = tf.nn.sigmoid(confidence)
        classes = tf.nn.sigmoid(classes)

        anchors = tf.tile(anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)

        x = tf.range(grid_shape[0], dtype=tf.float32)
        y = tf.range(grid_shape[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])

        strides = (img_size[0] // grid_shape[0], img_size[1] // grid_shape[1])
        box_centers = (box_centers + x_y_offset) * strides

        prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

        return prediction


    def get_layers_idx(self):
        return self.conv_layers_idx
