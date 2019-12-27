import numpy as np
import os
import tensorflow as tf
import cv2
from object_detection.utils import visualization_utils as vis_util
import sys

# ADD PATH TO FROZEN GRAPH
PATH_TO_FROZEN_GRAPH = '[path to your inference graph]/frozen_inference_graph.pb'

cap = cv2.VideoCapture(0)

# reads the frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# EDIT THE BELOW DICTIONARY ACCORDING TO YOUR LABEL MAP
# For Example,
# if your label_map.pbtxt was like:
# item {
#     id: 1
#     name: 'cat'
# }
# item {
#     id: 2
#     name: 'dog'
# }
# then your dictionary would look like this:
# category_index = {1: {'id': 1, 'name': 'cat'}, 2: {'id': 2, 'name': 'dog'}}
category_index = {1: {'id': 1, 'name': 'LABEL_1'}, 2: {'id': 2, 'name': 'LABEL_2'}}


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            # Read frame from camera
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                )
        # Display output
            cv2.imshow('Detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()
        sys.exit()
