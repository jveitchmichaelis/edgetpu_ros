# Edge TPU in ROS

This ROS package provides basic support for the Google Edge TPU, for example the Coral dev board or the USB accelerator.

Currently two nodes are provided which subscribe to an image topic and perform classification and detection. In theory you can run any (suitable) tensorflow network on the device, but you'll need to write your own node to do that. However, the vast majority of users will probably just want to detect or classify things.

You can find model files [here](https://coral.googlesource.com/edgetpu/+/refs/heads/release-chef/test_data/).

Getting this package going requires a little bit of fiddling. You need to be in a ROS environment with Python 3, because the Edge TPU API doesn't work with Python 2. As with most things involving ROS and Python, this can cause some problems. In particular, `cv_bridge` will cause you some trouble which you need to fix. This should probably be re-written to use the EdgeTPU C++ API.

This package has been tested with ROS Melodic, OpenCV 4.1 and Python 3.x. If you're going to go bleeding edge, you might as well make everything bleeding edge.

Things to maybe change:

- Switch to C++ for less installation pain
- Support for generic models

### Usage

Pretty simple, to run a detector:

``` xml
<launch>
  <node pkg="edge_tpu" name="detector" type="classify.py" output="screen">
    <param name="model_path" value="$(find edge_tpu)/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" />
    <param name="label_path" value="$(find edge_tpu)/models/coco_labels.txt" />
    <param name="threshold" type="int" value="0.75">
  </node>
</launch>
```

or a classifier:

``` xml
<launch>
  <node pkg="edge_tpu" name="detector" type="detect.py" output="screen">
    <param name="model_path" value="$(find edge_tpu)/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" />
    <param name="label_path" value="$(find edge_tpu)/models/coco_labels.txt" />
    <param name="threshold" type="int" value="0.75">
  </node>
</launch>
```

The nodes will listen to a topic called `input` which you should remap to your camera's image topic. The Edge API will do all the resizing for you, but the canned models are designed to work at around 300x300 so just be aware that if you send in a 5MP image, you'll lose quite a bit of resolution.

The detection node publishes a [Detection2DArray](http://docs.ros.org/api/vision_msgs/html/msg/Detection2D.html) topic (`detections`) which includes image crops for each bounding box. 

### Anaconda setup

Make a new conda environment:

``` bash
conda create -n ros-py3 python=3.7
conda activate ros-py3
```

You need to install the EdgeTPU API in your environment, the simplest way to do this is to clone the repository:

``` bash
git clone https://coral.googlesource.com/edgetpu
```

then build it:

``` bash
cd edgetpu
./build_package.sh
python setup.py sdist bdist_wheel

cd dist
pip install <edgetpu_version>-py3-none-any.whl
```

and test in a python REPL:

``` python
import edgetpu.classification.engine

```

If that's all good, then you're ready to get things going with ROS.

### ROS Setup

First, get some python packages:

`pip install rospkg catkin-pkg catkin-tools empy`

Also `opencv-contrib-python-headless` is useful for making a simple node to stream images from a camera. You might as well write your own capture node and this will get you the latest version of OpenCV.

Create a new workspace folder:

`mkdir -p edgetpu_ws/src`

Then, setup catkin to use Python 3. You can use your system python, if available, but I found Anaconda to be perfectly usable. You need to tell catkin where your python exe is, where the include directory is, and where the library is.

``` bash
catkin config -DPYTHON_EXECUTABLE=/home/josh/miniconda3/envs/ros-py3/bin/python \
              -DPYTHON_INCLUDE_DIR=/home/josh/miniconda3/envs/ros-py3/include/python3.5m \
              -DPYTHON_LIBRARY=/home/josh/miniconda3/envs/ros-py3/lib/libpython3.5m.so \
              -DSETUPTOOLS_DEB_LAYOUT=OFF
```

You also need to set `DSETUPTOOLS_DEB_LAYOUT` otherwise odd stuff seems to happen.

Then, clone the `cv_bridge` repository in your source folder. If you have OpenCV 3 built and in your path, then in theory everything is fine. The only issue you may come up against is that Numpy will shout at you. If it does, locate the offending file and check the #ifdef statement - just delete the bit that isn't for Python3.

If you're using OpenCV 4, then follow the instructions here to modify the package.

Run `catkin build` and fingers' crossed, everything should build correctly.
