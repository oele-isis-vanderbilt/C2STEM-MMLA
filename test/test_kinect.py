# Import Built-in
import pathlib
import os
import time
import logging

# Third-party Imports
import pytest
import cv2
import c2mmla

logger = logging.getLogger('')

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
GIT_ROOT = CWD.parent
TEST_KINECT_DATA = GIT_ROOT/'data'/'KinectData'/'OELE01'/'2022-10-05--11-52-55'
assert TEST_KINECT_DATA.exists()


@pytest.fixture
def rgb_cap():
    return cv2.VideoCapture(str(TEST_KINECT_DATA/'ColorStream.mp4'))


@pytest.fixture
def depth_cap():
    return cv2.VideoCapture(str(TEST_KINECT_DATA/'DepthStream.mp4'))


def test_kinect_data_same_frames(rgb_cap, depth_cap):

    rgb_total = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    depth_total = int(depth_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Num of frames: (COLOR={rgb_total}, DEPTH={depth_total})")

    assert ((rgb_total-depth_total)/depth_total) < 0.02

def test_kinect_data_show(rgb_cap, depth_cap):

    for i in range(1000):
        ret, frame = rgb_cap.read()
        ret, depth = depth_cap.read()

        cv2.imshow('color', frame)
        cv2.imshow('depth', depth)
        cv2.waitKey(1)


def test_video_step():
    # Make sure to have available data

    kinect = c2mmla.KinectNode(name="kinect", kinect_data_folder=TEST_KINECT_DATA, debug="step")

    kinect.prep()
    for i in range(50):
        kinect.step()

    kinect.teardown()
    kinect.shutdown()

def test_kinect_streaming():
    kinect = c2mmla.KinectNode(name="kinect", kinect_data_folder=TEST_KINECT_DATA, debug="stream")

    kinect.start()
    time.sleep(10)

    kinect.shutdown()
    kinect.join()
