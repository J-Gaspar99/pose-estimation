import argparse
import numpy as np
import cv2
import os
from utils import ucitaj_fajlove


def calibrate(images_path, file_name, width, height, cell_size, angle):

    print(angle)
    all_files = list()
    all_files = ucitaj_fajlove(images_path, ekstenzije={".png", ".jpg"})
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    single_object_points = np.zeros((width*height, 3), np.float32)
    for i in range(height):
        for j in range(width):
            single_object_points[i*width + j, :] = np.array(
                [j*cell_size, i*cell_size, 0], dtype=np.float32)
    object_points = []  # 3d point in real world space
    imgage_points = []
    i = 0
    for image in all_files:
        i+=1
        img = cv2.imread(image)
        # using cv2.getRotationMatrix2D() to get the rotation matrix
        #height, width = img.shape[0:2]
        #print(height,width)
        #print(img)
        #rotationMatrix = cv2.getRotationMatrix2D(center = (width/2, height/2), angle=float(angle), scale=1)
        #Rotated_image = cv2.warpAffine(img, rotationMatrix, (width, height))

        #cv2.imshow("slika",image)

        # rotate the image using cv2.warpAffine
        #rotated_image = cv2.warpAffine(img, M=rotationMatrix, dsize=(width, height))
        #print(i)
        #print(rotated_image)

       # height1, width1 = rotated_image.shape[0:2]
       # print(height1, width1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(single_object_points)
            imgage_points.append(corners2)
            img = cv2.drawChessboardCorners(
                img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, imgage_points, gray.shape[::-1], None, None)
    mean_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(
            single_object_points, rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgage_points[i],
                         imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error/len(object_points)))

    file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    file.write("distortion", dist)
    file.write("intrinsic", mtx)
    file.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program za kalibraciju kamere i čuvanje unutrašnjih parametara kamere',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-p", "--path", help="Putanja u kojoj se nalaze slike u png/jpg formatu")
    parser.add_argument("-f", "--file_name",
                        help="Ime rezultujućeg kalibracionog fajla")
    parser.add_argument("-c", "--cell_size",
                        help="Veličina ćelije u milimetrima", default=30)
    parser.add_argument("-g", "--grid_size",
                        help="Veličina table", default="8x5")
    parser.add_argument("-r", "--rotation_angle",
                        help="Rotacioni ugao", default="0")
    options = parser.parse_args()
    (w, h) = options.grid_size.split('x')
    calibrate(options.path, options.file_name,int(w), int(h), float(options.cell_size), options.rotation_angle)
