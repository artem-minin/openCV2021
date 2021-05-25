#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

'''
Deeplabv3 person running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 deeplabv3_person_256.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob taken from the great PINTO zoo

git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob

'''

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)#'models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob'
parser.add_argument("-nn", "--nn_model", help="select camera input source for inference", default='models/class_model_mobilenet_v3_small_data3_class_weights_512x512_without_softmax_6shaves.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input 
nn_path = args.nn_model 

nn_shape = 256 #size of square image 
if '513' in nn_path:
    nn_shape = 513
if '512' in nn_path:
    nn_shape = 512
def decode_deeplabv3p(output_tensor):
    #["Soil":BROWN,"Clover:RED","Broadleaf:PURPLE","Grass:ORANGE"]
    class_colors = [[166, 86, 40], [228, 26, 28], [155, 126, 184], [255, 127, 0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    #save output colors. 
    
    return cv2.addWeighted(frame,1, output_colors,0.2,0)



# Start defining a pipeline
pipeline = dai.Pipeline()

if '513' in nn_path:
    nn_shape = 513
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
if '512' in nn_path:
    nn_shape = 512
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_3)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape,nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(40)
####################ROI##########################################
stepSize = 0.05
newConfig = False
# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

increaseROI = 0
topLeft = dai.Point2f(0.4-increaseROI, 0.4-increaseROI)
bottomRight = dai.Point2f(0.6+increaseROI, 0.6+increaseROI)

spatialLocationCalculator.setWaitForConfigInput(False)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)
spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

#############################################
# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False) #Nueral network input
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)#Neural network output 
# Output queue will be used to get the depth frames from the outputs defined above
############################ROI###################################################
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
####################################################################################
start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False
while True:
    ####################################ROI##################################
    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

    depthFrame = inDepth.getFrame()
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

    spatialData = spatialCalcQueue.get().getSpatialLocations()
    for depthData in spatialData:
        roi = depthData.config.roi
        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
        xmin = int(roi.topLeft().x)
        ymin = int(roi.topLeft().y)
        xmax = int(roi.bottomRight().x)
        ymax = int(roi.bottomRight().y)

        fontType = cv2.FONT_HERSHEY_TRIPLEX
        color = (255, 255, 255)
        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
        cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
        cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
        cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)
    # Show the frame
    cv2.imshow("depth", depthFrameColor)
    #Move ROI
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        if topLeft.y - stepSize >= 0:
            topLeft.y -= stepSize
            bottomRight.y -= stepSize
            newConfig = True
    elif key == ord('a'):
        if topLeft.x - stepSize >= 0:
            topLeft.x -= stepSize
            bottomRight.x -= stepSize
            newConfig = True
    elif key == ord('s'):
        if bottomRight.y + stepSize <= 1:
            topLeft.y += stepSize
            bottomRight.y += stepSize
            newConfig = True
    elif key == ord('d'):
        if bottomRight.x + stepSize <= 1:
            topLeft.x += stepSize
            bottomRight.x += stepSize
            newConfig = True

    if newConfig:
        config.roi = dai.Rect(topLeft, bottomRight)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)
        newConfig = False

    #######################################ROI#############################################
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None: ########Neural Network input 
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
        frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    if in_nn is not None: ######Nueral Network Prediciton 
        # print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                dims = layer.dims[::-1] # reverse dimensions
                print(f"dims: {dims}")
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getLayerInt32(layers[0].name)
        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)

        output_colors = decode_deeplabv3p(lay1)
        labeled_output = output_colors
        cv2.putText(labeled_output, "Soil: BROWN, Clover: RED, Broadleaf: PURPLE , Grass: ORANGE", (2, labeled_output.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 0, 0))
        cv2.imshow("seg_mask",labeled_output)
        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            cv2.imshow("nn_input", frame)#was frame
            #cv2.imshow("seg_mask",output_colors)
    
    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()


    if cv2.waitKey(1) == ord('q'):
        break
