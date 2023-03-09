import argparse
import numpy as np
import cv2
import os
from utils import ucitaj_unutrasnje_parametre

def undistort_video(input_video, calib_file, output_video):
    cam_mat, dist_coeffs = ucitaj_unutrasnje_parametre(calib_file)
    try:
        video_input = cv2.VideoCapture(input_video)
    except:
        print("Nisam mogao da otvorim video")
        return
    if not video_input.isOpened():
        print("Nisam mogao da otvorim video")
        return
    width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video_input.get(cv2.CAP_PROP_FPS))
    fourcc = int(video_input.get(cv2.CAP_PROP_FOURCC))
    video_out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(
        cam_mat, dist_coeffs, None, cam_mat, (width, height),cv2.CV_16SC2)
    cv2.namedWindow('Undistorted', cv2.WINDOW_KEEPRATIO)
    while True:
        ret, frame = video_input.read()
        if not ret:
            break
        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        cv2.imshow("Undistorted", np.hstack((frame, undistorted)))
        cv2.waitKey(int(1000.0/2/fps))
        video_out.write(undistorted)
    video_out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program za uklanjanje izoblicenja iz videa',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", "--video", help="Ime ulaznog videa", required=True)
    parser.add_argument("-b", "--calib_file",
                        help="Ime ulaznog kalibracionog fajla. Kalibracioni fajl mora da sadrži takove <intrinsic> i <distortion>", required=True)
    parser.add_argument("-o", "--output",
                        help="Ime izlaznog videa", required=True)
    options = parser.parse_args()
    undistort_video(options.video, options.calib_file, options.output)
