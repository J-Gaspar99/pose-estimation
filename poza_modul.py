import argparse
import cv2
import mediapipe as mp
import time

from google.protobuf.json_format import MessageToDict
from numpy import linalg as LA
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def ucitaj_fajlove(folder, sort=True, ekstenzije=None):
    all_files = list()
    for f in os.listdir(folder):
        if ekstenzije == None or (os.path.splitext(f)[-1] in ekstenzije):
            all_files.append(os.path.join(folder, f))
    if sort:
        all_files.sort()
    return all_files


def sacuvaj_pozu(pose, i, filename_template):
    name = filename_template.format(num=i)
    #print(pose)
    poza = {"people": [{"pose_keypoints_3d":str(pose)}]}
    json_object = json.dumps(poza, indent=2)
    with open(name, "w") as outfile:
        outfile.write(json_object)

def calculateAngle(a, b, c):
                #print("trazi se ugao izmedju:"+str(a)+str(b)+str(c))
                a = np.array(a[1])
                b = np.array(b[1])
                c = np.array(c[1])
                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                if angle > 180.0:
                    angle = 360 - angle
                return angle

class poseDetector():
    def __init__(self,
        mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        enable_segmentation = False,
        smooth_segmentation = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5):

        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.mpPose_LANDMARKS = mp.solutions.pose.PoseLandmark
        #self.mpPose_WORLD_LANDMARKS = self.mpPose.POSE_WORLD_LANDMARKS


        self.pose = self.mpPose.Pose( self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)

    def findPose(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)
            if self.results.pose_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            return img, self.results.pose_world_landmarks

    def getPosition(self, img):
        keypoints = []
        coor_list = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = lm.x*w, lm.y*h

                keypoints.append({
                        'ID': id,
                        'X': lm.x,
                        'Y': lm.y,
                        'Z': lm.z,
                        'Visibility': lm.visibility,
                    })

                coor_list.append([id,(lm.x,lm.y,lm.z)])

                sacuvaj_pozu(keypoints,id,"izlaz/json")

            return coor_list

# 5,2 obrve
# 29,30 pete
    def odredi_visinu(self, coor_list):
        #[1,0]
        #[NAZIV,(x,y)]
        tackaA = ((coor_list[5][1][0]+coor_list[2][1][0])/2,((coor_list[5][1][1]+coor_list[2][1][1])/2))
        tackaB = ((coor_list[29][1][0]+coor_list[30][1][0])/2,((coor_list[29][1][1]+coor_list[30][1][1])/2))
        #LA.norm(tackaA+tackaB)
        visina = (round(np.sqrt(((tackaA[1]-tackaB[1])**2)/3),2))
        return  visina

    def klasifikujPozu(self, landmarks, output_image, display=False):

            # Initialize the label of the pose. It is not known at this stage.
            label = 'Unknown Pose'
            color = (0, 0, 255)
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist) #180
            angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)  #180

            angle3 = calculateAngle(right_elbow, right_shoulder, right_hip) #45;90
            angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)    #90;45

            angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
            angle6 = calculateAngle(left_shoulder, left_hip, left_knee)

            angle7 = calculateAngle(right_hip, right_knee, right_ankle) #180
            angle8 = calculateAngle(left_hip, left_knee, left_ankle)    #180


            # za M pozu
            if angle1 > 160 and angle1 < 200 and angle2 > 160 and angle2 < 200:
                    
                if angle3 > 30 and angle3 < 50 and angle4 > 70 and angle4 < 110:
                    label = 'M Pose'

                if angle4 > 30 and angle4 < 50 and angle3 > 70 and angle3 < 110:
                    label = 'M Pose'


            if label != 'Unknown Pose':
                color = (100, 120, 175)

            cv2.putText(output_image, label, (500, 50), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            cv2.rectangle(output_image, (0, 0), (100, 255), (255, 255, 255), -1)

            cv2.putText(output_image, 'ID', (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(1), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(2), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(3), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(4), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(5), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(6), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(7), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(8), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

            cv2.putText(output_image, 'Angle', (40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle1)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle2)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle3)), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle4)), (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle5)), (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle6)), (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle7)), (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)
            cv2.putText(output_image, str(int(angle8)), (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                        cv2.LINE_AA)

            if display:

                plt.figure(figsize=[10, 10])
                plt.imshow(output_image[:, :, ::-1]);
                plt.title("Output Image");
                plt.axis('off');

            else:

                return output_image, label





def main(putanja1):
        #print (putanja)

        pd = poseDetector()
        pTime = 0
        cap = cv2.VideoCapture(putanja1)

        while True:
            success, img = cap.read()
            i=0
            if (success):

                img, w_landmarks = pd.findPose(img)
               # sacuvaj_pozu(w_landmarks, i, "izlaz/sample1.json")

                a = pd.getPosition(img)
                #print(a)

                data = {
                        "frame_number": i,
                        "frame_data": a
                    }


                i=i+1


                print("pozicija:"+str(a))
                if(a!=None):
                    #print(a)
                    visina = pd.odredi_visinu(a)
                    #print(visina)
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    poza = pd.klasifikujPozu(landmarks=a, output_image=img)

                    #with open("izlaz/sample.json", 'a') as fl:
                        #for idx, coords in enumerate(landmarks):
                    #        coords_dict = MessageToDict(a)
                    #        fl.write(json.dumps(coords_dict, indent=2, separators=(',', ': ')))

                    #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    #(255, 255, 0), 3)
                    cv2.putText(img, str(visina+1), (270, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 0), 3)

                    cv2.imshow("Image", img)
                    cv2.waitKey(1)
                else:
                    cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()
                break



            #print(str(pd.mpPose_WORLD_LANDMARKS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program za prepoznavanje poze, i odredjivanje visine',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--putanja", help="Putanje do videa", nargs="+", required=True)

    #parser.add_argument("-o", "--output",
    #                    help="Ime rezultujucih JSON fajlova", default="poze_{num:05d}.json")
    options = parser.parse_args()

    main(options.putanja[0])
