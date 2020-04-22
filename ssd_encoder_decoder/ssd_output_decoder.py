'''
Includes:
* Functions to decode and filter raw SSD model output. These are only needed if the
  SSD model does not have a `DecodeDetections` layer.
* Functions to perform greedy non-maximum suppression

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np

from bounding_box_utils.bounding_box_utils import iou, convert_coordinates

# import tensorflow as tf
# import torch 
# import torch.nn.functional as F

def yolact_nms(boxes, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        print("ENTERING YOLACT NMS")
        max_num_detections = 100

        # scores, idx = scores.sort(1, descending=True)
        print("inputted scores: ", scores)

        idx = np.argsort(scores)
        scores = np.sort(scores)
        
        
        print("sorted scores: ", scores)
        print("sorted idx: ", idx)
    

        # FIX THIS
        idx = idx[:top_k]
        # idx = idx[:top_k].contiguous()
        print("top_k idx: ", idx)

    
        scores = scores[:top_k]
        print("top_k scores: ", scores)

        # # FIX THIS
        # num_classes, num_dets = idx.size()
        # print("num_classes: ", num_classes)
        # print("num_dets: ", num_dets)


        # # FIX THIS
        # boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        # print("boxes: ", boxes)


        print("boxes: ", boxes)

        # FIX THIS
        jac = iou(boxes, boxes)
        print("jac: ", jac)
        print("jac_shape: ", jac.shape)
        
        # jac.triu_(diagonal=1)
        jac = np.triu(jac, k=1)
        print("jac_triu: ", jac)


        # jac_max, _ = jac.max(dim=1)
        jac_max = jac.max(1)
        print("jac_max: ", jac_max)


        # Now just filter out the ones higher than the threshold
        keep = (jac_max <= iou_threshold)
        print("iou_threshold: ", iou_threshold)
        print("keep: ", keep)


        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        
        # to use, add self to input parameters
        # if second_threshold:
        #     keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        # classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        # classes = classes[keep]

        boxes = boxes[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        # scores, idx = scores.sort(0, descending=True)
        
        idx = np.argsort(scores)        
        scores = np.sort(scores)

        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]

        # classes = classes[idx]
        boxes = boxes[idx]

        print("boxes: ", boxes)
        print("boxes shape: ", boxes.shape)
        
        print("scores: ", scores)
        print("scores shape: ", scores.shape)

        # scores = np.array(scores)
        # print("scores: ", scores)
        # print("scores shape: ", scores.shape)


        # print(type(boxes),type(scores))

        scores = scores.reshape(scores.shape[0], -1)

        maxima = np.hstack((scores, boxes))
        print("maxima:\n", maxima)



        print("EXITING YOLACT NMS")




        return maxima

def yolact_nms_decoder(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                
                # CHANGING NMS FUNCTION

                print("threshold_met: ", threshold_met)
                print("threshold_met shape: ", threshold_met.shape)
                

                boxes = threshold_met[:,-4:]
                scores = threshold_met[:,0]
                print("boxes: ", boxes)
                print("boxes shape: ", boxes.shape)
                print("scores: ", scores)
                print("scores shape: ", scores.shape)

                # max_output_size = None
                
                              
                # maxima = yolact_nms(boxes, scores)
                # print("maxima: ", maxima)
                # print("maxima shape: ", maxima.shape)

                maxima = yolact_nms(boxes, scores)
                # print("boxes: ", boxes)
                # print("boxes shape: ", boxes.shape)
                # print("scores: ", scores)
                # print("scores shape: ", scores.shape)



                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded



def soft_nms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限 overlap threshold
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """
    print("ENTERING SOFT NMS")
    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)
    print("dets: ", dets)


    # the order of boxes coordinate is [y1,x1,y2,x2]

    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    print("x1:" , x1)
    print("x2:" , x2)
    print("y1:" , y1)
    print("y2:" , y2)
    print("sc: ", sc)
    conf = sc.copy()
    scores = sc
    print("scores:" , scores)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    print("areas:" , areas)

    for i in range(N):
        print("\nSOFT NMS LOOP ", i)
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        print("tBD", i, ": " , tBD)

        tscore = scores[i].copy()
        print("tscore", i, ": " , tscore)

        tarea = areas[i].copy()
        print("tarea", i, ": " , tarea)
        
        pos = i + 1
        print("pos:" , pos)

        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        print("maxscore:" , maxscore)
        print("maxpos:" , maxpos)
        print("tscore:" , tscore)
        
        print("dets: ", dets)


        # SORT THE DETECTIONS
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]
        print("dets: ", dets)



        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)
        print("ovr ", i, ": " , ovr)


        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0
        print("scores[pos:] ", scores[pos:])
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    print("dets[:, 4]: ", dets[:, 4])
    
    print("scores: ", scores)
    print("scores.shape: ", scores.shape)


    inds = dets[:, 4][scores > thresh]
    print("inds: ", inds)    
    inds = inds.astype(int)
    print("inds: ", inds)


    # GETTING SCORES

    # conf = conf[inds]
    # conf = conf.reshape(conf.shape[0], -1)
    # print("conf: ", conf)
    # print("conf.shape: ", conf.shape)

    scores = scores[inds]
    scores = scores.reshape(scores.shape[0], -1)
    print("scores: ", scores)
    print("scores.shape: ", scores.shape)







    dets = dets[inds, :4]
    print("dets: ", dets)
    print("dets_shape: ", dets.shape)


    keep = np.hstack((scores, dets))
    print("keep: ", keep)
    print("keep_shape: ", keep.shape)

    print("EXITING SOFT NMS")

    return keep

def soft_nms_decoder(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                
                # CHANGING NMS FUNCTION

                # print("threshold_met: ", threshold_met)
                # print("threshold_met shape: ", threshold_met.shape)
                

                boxes = threshold_met[:,-4:]
                scores = threshold_met[:,0]
                print("boxes: ", boxes)
                print("boxes shape: ", boxes.shape)
                print("scores: ", scores)
                print("scores shape: ", scores.shape)

                # max_output_size = None
                


                maxima = soft_nms(boxes, scores)
                # print("boxes: ", boxes)
                # print("boxes shape: ", boxes.shape)
                # print("scores: ", scores)
                # print("scores shape: ", scores.shape)



                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded




def mal_nms(boxes, overlapThresh=0.45):
    print("STARTING Mal NMS")
	# if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
	# initialize the list of picked indexes	
    pick = []
	# boxes [confidence, xmin, ymin, xmax, ymax]
	# grab the coordinates of the bounding boxes
    print("boxes: ", boxes)
    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    print("x1:" , x1)
    print("x2:" , x2)
    print("y1:" , y1)
    print("y2:" , y2)

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    print("area:" , area)

    probs = boxes[:,0]

    idxs = np.argsort(probs)
    print("probs: ", probs)
	# keep looping while some indexes still remain in the indexes
	# list
    counter = 1
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
        print("START", counter)
        print("idxs: ", idxs)
        last = len(idxs) - 1
        i = idxs[last]
        print("i: ", i)
        pick.append(i)
        print("pick: ", pick)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        print("xx1: ", xx1)
        print("yy1: ", yy1)
        print("xx2: ", xx2)
        print("yy2: ", yy2)

		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        print("w:" , w)
        h = np.maximum(0, yy2 - yy1 + 1)
        print("h:" , h)
		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        print("overlap: ", overlap)
		# delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        print("END", counter, "\n")
        counter += 1
	# return only the bounding boxes that were picked using the
	# integer data type
	# return boxes[pick]
    print("EXITING Mal NMS")
    return boxes[pick]
    # return boxes[pick].astype("int")

def mal_nms_decoder(y_pred,
                      confidence_thresh=0.3,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates
    print("y_pred_decoded_raw shape: ", y_pred_decoded_raw.shape)
    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates
    print("n_classes: ", n_classes)
    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        print("batch_item: ", batch_item)
        print("batch_item shape: ", batch_item.shape)
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            print("threshold_met: ", threshold_met)
            print("threshold_met shape: ", threshold_met.shape)
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
 

                maxima_mal_nms = mal_nms(threshold_met, iou_threshold) # ...perform NMS on them.
                print("maxima_mal_nms: ", maxima_mal_nms)
                print("maxima_mal_nms shape: ", maxima_mal_nms.shape)



                maxima_output = np.zeros((maxima_mal_nms.shape[0], maxima_mal_nms.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima_mal_nms # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded



def greedy_nms(y_pred_decoded, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    Perform greedy non-maximum suppression on the input boxes.

    Greedy NMS works by selecting the box with the highest score and
    removing all boxes around it that are too close to it measured by IoU-similarity.
    Out of the boxes that are left over, once again the one with the highest
    score is selected and so on, until no boxes with too much overlap are left.

    Arguments:
        y_pred_decoded (list): A batch of decoded predictions. For a given batch size `n` this
            is a list of length `n` where each list element is a 2D Numpy array.
            For a batch item with `k` predicted boxes this 2D Numpy array has
            shape `(k, 6)`, where each row contains the coordinates of the respective
            box in the format `[class_id, score, xmin, xmax, ymin, ymax]`.
            Technically, the number of columns doesn't have to be 6, it can be
            arbitrary as long as the first four elements of each row are
            `xmin`, `xmax`, `ymin`, `ymax` (in this order) and the last element
            is the score assigned to the prediction. Note that this function is
            agnostic to the scale of the score or what it represents.
        iou_threshold (float, optional): All boxes with a Jaccard similarity of
            greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score.
        coords (str, optional): The coordinate format of `y_pred_decoded`.
            Can be one of the formats supported by `iou()`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        The predictions after removing non-maxima. The format is the same as the input format.
    '''
    y_pred_decoded_nms = []
    for batch_item in y_pred_decoded: # For the labels of each batch item...
        boxes_left = np.copy(batch_item)
        maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
        while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
            maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
            print("Maximum index:", maximum_index, maximum_index.shape())
            maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
            print("Maximum box:", maximum_box, maximum_box.shape())
            maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
            boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
            if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
            similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
            boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        y_pred_decoded_nms.append(np.array(maxima))

    return y_pred_decoded_nms

def _greedy_nms(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections()`.
    '''
    print("ENTERING GREEDY NMS")
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    counter = 1
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        print("START loop greedy", counter)
        print("boxes_left:" , boxes_left)
        print("boxes_left shape: ", boxes_left.shape)
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        print("Maximum index:", maximum_index, maximum_index.shape)
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        print("Maximum box:", maximum_box, maximum_box.shape)
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,1:], maximum_box[1:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        print("END loop greedy", counter, "\n")
        counter += 1
    print("EXITING GREEDY NMS")
    return np.array(maxima)

def _greedy_nms2(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function in `decode_detections_fast()`.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)

def decode_detections(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=200,
                      input_coords='centroids',
                      normalize_coords=True,
                      img_height=None,
                      img_width=None,
                      border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `SSDInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item. This procedure follows the original Caffe implementation.
    For a slightly different and more efficient alternative to decode raw model output that performs
    non-maximum suppresion globally instead of per class, see `decode_detections_fast()` below.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                
                # CHANGING NMS FUNCTION
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                

                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded


def decode_detections_fast(y_pred, 
                           confidence_thresh=0.3,
                           iou_threshold=0.45,
                           top_k='all',
                           input_coords='centroids',
                           normalize_coords=True,
                           img_height=None,
                           img_width=None,
                           border_pixels='half'):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `enconde_y()` takes as input).

    Optionally performs confidence thresholding and greedy non-maximum suppression after the decoding stage.

    Note that the decoding procedure used here is not the same as the procedure used in the original Caffe implementation.
    For each box, the procedure used here assigns the box's highest confidence as its predicted class. Then it removes
    all boxes for which the highest confidence is the background class. This results in less work for the subsequent
    non-maximum suppression, because the vast majority of the predictions will be filtered out just by the fact that
    their highest confidence is for the background class. It is much more efficient than the procedure of the original
    implementation, but the results may also differ.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in any positive
            class required for a given box to be considered a positive prediction. A lower value will result
            in better recall, while a higher value will result in better precision. Do not use this parameter with the
            goal to combat the inevitably many duplicates that an SSD will produce, the subsequent non-maximum suppression
            stage will take care of those.
        iou_threshold (float, optional): `None` or a float in [0,1]. If `None`, no non-maximum suppression will be
            performed. If not `None`, greedy NMS will be performed after the confidence thresholding stage, meaning
            all boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score.
        top_k (int, optional): 'all' or an integer with number of highest scoring predictions to be kept for each batch item
            after the non-maximum suppression stage. If 'all', all predictions left after the NMS stage will be kept.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 6)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, xmin, xmax, ymin, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the classes from one-hot encoding to their class ID
    y_pred_converted = np.copy(y_pred[:,:,-14:-8]) # Slice out the four offset predictions plus two elements whereto we'll write the class IDs and confidences in the next step
    y_pred_converted[:,:,0] = np.argmax(y_pred[:,:,:-12], axis=-1) # The indices of the highest confidence values in the one-hot class vectors are the class ID
    y_pred_converted[:,:,1] = np.amax(y_pred[:,:,:-12], axis=-1) # Store the confidence values themselves, too

    # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    if input_coords == 'centroids':
        y_pred_converted[:,:,[4,5]] = np.exp(y_pred_converted[:,:,[4,5]] * y_pred[:,:,[-2,-1]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
        y_pred_converted[:,:,[4,5]] *= y_pred[:,:,[-6,-5]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
        y_pred_converted[:,:,[2,3]] *= y_pred[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
        y_pred_converted[:,:,[2,3]] += y_pred[:,:,[-8,-7]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_converted[:,:,2:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_converted[:,:,[2,3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[4,5]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,2:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_converted = convert_coordinates(y_pred_converted, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_converted[:,:,2:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_converted[:,:,[2,4]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_converted[:,:,[3,5]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_converted[:,:,2:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `coords`. Supported values are 'minmax', 'corners' and 'centroids'.")

    # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_converted[:,:,[2,4]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_converted[:,:,[3,5]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
    y_pred_decoded = []
    for batch_item in y_pred_converted: # For each image in the batch...
        boxes = batch_item[np.nonzero(batch_item[:,0])] # ...get all boxes that don't belong to the background class,...
        boxes = boxes[boxes[:,1] >= confidence_thresh] # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
        if iou_threshold: # ...if an IoU threshold is set...
            boxes = _greedy_nms2(boxes, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on the remaining boxes.
        if top_k != 'all' and boxes.shape[0] > top_k: # If we have more than `top_k` results left at this point...
            top_k_indices = np.argpartition(boxes[:,1], kth=boxes.shape[0]-top_k, axis=0)[boxes.shape[0]-top_k:] # ...get the indices of the `top_k` highest-scoring boxes...
            boxes = boxes[top_k_indices] # ...and keep only those boxes...
        y_pred_decoded.append(boxes) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded

################################################################################################
# Debugging tools, not relevant for normal use
################################################################################################

# The functions below are for debugging, so you won't normally need them. That is,
# unless you need to debug your model, of course.

def decode_detections_debug(y_pred,
                            confidence_thresh=0.01,
                            iou_threshold=0.45,
                            top_k=200,
                            input_coords='centroids',
                            normalize_coords=True,
                            img_height=None,
                            img_width=None,
                            variance_encoded_in_target=False,
                            border_pixels='half'):
    '''
    This decoder performs the same processing as `decode_detections()`, but the output format for each left-over
    predicted box is `[box_id, class_id, confidence, xmin, ymin, xmax, ymax]`.

    That is, in addition to the usual data, each predicted box has the internal index of that box within
    the model (`box_id`) prepended to it. This allows you to know exactly which part of the model made a given
    box prediction; in particular, it allows you to know which predictor layer made a given prediction.
    This can be useful for debugging.

    Arguments:
        y_pred (array): The prediction output of the SSD model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 predicted coordinate offsets, 4 anchor box coordinates, 4 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        input_coords (str, optional): The box coordinate format that the model outputs. Can be either 'centroids'
            for the format `(cx, cy, w, h)` (box center coordinates, width, and height), 'minmax' for the format
            `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxex, but not the other.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 7)` where each row is a box prediction for
        a non-background class for the respective image in the format `[box_id, class_id, confidence, xmin, ymin, xmax, ymax]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-8]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    if input_coords == 'centroids':
        if variance_encoded_in_target:
            # Decode the predicted box center x and y coordinates.
            y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] + y_pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
        else:
            # Decode the predicted box center x and y coordinates.
            y_pred_decoded_raw[:,:,[-4,-3]] = y_pred_decoded_raw[:,:,[-4,-3]] * y_pred[:,:,[-6,-5]] * y_pred[:,:,[-4,-3]] + y_pred[:,:,[-8,-7]]
            # Decode the predicted box width and heigt.
            y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]] * y_pred[:,:,[-2,-1]]) * y_pred[:,:,[-6,-5]]
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='centroids2corners')
    elif input_coords == 'minmax':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-3]] *= np.expand_dims(y_pred[:,:,-7] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-2,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-6], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
        y_pred_decoded_raw = convert_coordinates(y_pred_decoded_raw, start_index=-4, conversion='minmax2corners')
    elif input_coords == 'corners':
        y_pred_decoded_raw[:,:,-4:] *= y_pred[:,:,-4:] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
        y_pred_decoded_raw[:,:,[-4,-2]] *= np.expand_dims(y_pred[:,:,-6] - y_pred[:,:,-8], axis=-1) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
        y_pred_decoded_raw[:,:,[-3,-1]] *= np.expand_dims(y_pred[:,:,-5] - y_pred[:,:,-7], axis=-1) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
        y_pred_decoded_raw[:,:,-4:] += y_pred[:,:,-8:-4] # delta(pred) + anchor == pred for all four coordinates
    else:
        raise ValueError("Unexpected value for `input_coords`. Supported input coordinate formats are 'minmax', 'corners' and 'centroids'.")

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that

    if normalize_coords:
        y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
        y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: For each batch item, prepend each box's internal index to its coordinates.

    y_pred_decoded_raw2 = np.zeros((y_pred_decoded_raw.shape[0], y_pred_decoded_raw.shape[1], y_pred_decoded_raw.shape[2] + 1)) # Expand the last axis by one.
    y_pred_decoded_raw2[:,:,1:] = y_pred_decoded_raw
    y_pred_decoded_raw2[:,:,0] = np.arange(y_pred_decoded_raw.shape[1]) # Put the box indices as the first element for each box via broadcasting.
    y_pred_decoded_raw = y_pred_decoded_raw2

    # 4: Apply confidence thresholding and non-maximum suppression per class

    n_classes = y_pred_decoded_raw.shape[-1] - 5 # The number of classes is the length of the last axis minus the four box coordinates and minus the index

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[0, class_id + 1, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 6]` and...
            threshold_met = single_class[single_class[:,1] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms_debug(threshold_met, iou_threshold=iou_threshold, coords='corners', border_pixels=border_pixels) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = maxima[:,0] # Write the box index to the first column...
                maxima_output[:,1] = class_id # ...and write the class ID to the second column...
                maxima_output[:,2:] = maxima[:,1:] # ...and write the rest of the maxima data to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        pred = np.concatenate(pred, axis=0)
        if pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
            top_k_indices = np.argpartition(pred[:,2], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
            pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded

def _greedy_nms_debug(predictions, iou_threshold=0.45, coords='corners', border_pixels='half'):
    '''
    The same greedy non-maximum suppression algorithm as above, but slightly modified for use as an internal
    function for per-class NMS in `decode_detections_debug()`. The difference is that it keeps the indices of all
    left-over boxes for each batch item, which allows you to know which predictor layer predicted a given output
    box and is thus useful for debugging.
    '''
    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,1]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...
        similarities = iou(boxes_left[:,2:], maximum_box[2:], coords=coords, mode='element-wise', border_pixels=border_pixels) # ...compare (IoU) the other left over boxes to the maximum box...
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
    return np.array(maxima)

def get_num_boxes_per_pred_layer(predictor_sizes, aspect_ratios, two_boxes_for_ar1):
    '''
    Returns a list of the number of boxes that each predictor layer predicts.

    `aspect_ratios` must be a nested list, containing a list of aspect ratios
    for each predictor layer.
    '''
    num_boxes_per_pred_layer = []
    for i in range(len(predictor_sizes)):
        if two_boxes_for_ar1:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * (len(aspect_ratios[i]) + 1))
        else:
            num_boxes_per_pred_layer.append(predictor_sizes[i][0] * predictor_sizes[i][1] * len(aspect_ratios[i]))
    return num_boxes_per_pred_layer

def get_pred_layers(y_pred_decoded, num_boxes_per_pred_layer):
    '''
    For a given prediction tensor decoded with `decode_detections_debug()`, returns a list
    with the indices of the predictor layers that made each predictions.

    That is, this function lets you know which predictor layer is responsible
    for a given prediction.

    Arguments:
        y_pred_decoded (array): The decoded model output tensor. Must have been
            decoded with `decode_detections_debug()` so that it contains the internal box index
            for each predicted box.
        num_boxes_per_pred_layer (list): A list that contains the total number
            of boxes that each predictor layer predicts.
    '''
    pred_layers_all = []
    cum_boxes_per_pred_layer = np.cumsum(num_boxes_per_pred_layer)
    for batch_item in y_pred_decoded:
        pred_layers = []
        for prediction in batch_item:
            if (prediction[0] < 0) or (prediction[0] >= cum_boxes_per_pred_layer[-1]):
                raise ValueError("Box index is out of bounds of the possible indices as given by the values in `num_boxes_per_pred_layer`.")
            for i in range(len(cum_boxes_per_pred_layer)):
                if prediction[0] < cum_boxes_per_pred_layer[i]:
                    pred_layers.append(i)
                    break
        pred_layers_all.append(pred_layers)
    return pred_layers_all
