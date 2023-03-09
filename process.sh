# #!/usr/bin/env bash
CAM1="950122060409"
CAM2="104122061553"
CAM3="950122061749"
VID="vid_00002.mp4"
CALIB="2"

echo "Formiranje direktorijuma"

mkdir izlaz/
mkdir -p izlaz/$CAM1/json
mkdir -p izlaz/$CAM2/json
mkdir -p izlaz/$CAM3/json
mkdir izlaz/rekonstruisan

echo "Kalibracija kamera1"
python kalibracija.py -p calib/$CAM1 -f izlaz/$CAM1/kalibracija.xml

echo "Kalibracija kamera2"
python kalibracija.py -p calib/$CAM2 -f izlaz/$CAM2/kalibracija.xml -r 60

echo "Kalibracija kamera3"
python kalibracija.py -p calib/$CAM3 -f izlaz/$CAM3/kalibracija.xml

echo "Odredjujem pozicije kamera u prostoru"

python pozicije_kamera.py   -p pozicije/$CALIB/$CAM1 \
                            -b izlaz/$CAM1/kalibracija.xml \
                            -f izlaz/$CAM1/pozicija.xml \
                            -c 45

python pozicije_kamera.py   -p pozicije/$CALIB/$CAM2 \
                            -b izlaz/$CAM2/kalibracija.xml \
                            -f izlaz/$CAM2/pozicija.xml \
                            -c 45

python pozicije_kamera.py   -p pozicije/$CALIB/$CAM3 \
                            -b izlaz/$CAM3/kalibracija.xml \
                            -f izlaz/$CAM3/pozicija.xml \
                            -c 45
echo "Uklanjam izoblicenje iz videa"
python undistort_video.py   -v pokret/$CAM1/$VID \
                            -b izlaz/$CAM1/kalibracija.xml \
                            -o izlaz/$CAM1/undistorted.mp4

python undistort_video.py   -v pokret/$CAM2/$VID \
                            -b izlaz/$CAM2/kalibracija.xml \
                            -o izlaz/$CAM2/undistorted.mp4

python undistort_video.py   -v pokret/$CAM3/$VID \
                            -b izlaz/$CAM3/kalibracija.xml \
                            -o izlaz/$CAM3/undistorted.mp4

python poza_modul.py  -p izlaz/$CAM1/undistorted.mp4  \
                      #-o izlaz/$CAM1/json/a.json


python poza_modul.py  -p izlaz/$CAM2/undistorted.mp4 \
                     # -o izlaz/$CAM2/json/poze_{num:05d}.json


python poza_modul.py  -p izlaz/$CAM3/undistorted.mp4 \
                     # -o izlaz/$CAM3/json/poze_{num:05d}.json


cd mediapipe_pose_3d_reconstruction && cd bodypose3d && python bodypose3d.py

sleep 3
