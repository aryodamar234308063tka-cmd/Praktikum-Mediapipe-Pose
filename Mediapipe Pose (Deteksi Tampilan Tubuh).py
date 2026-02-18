import cv2
import mediapipe as mp
mpPose = mp.solutions.pose #inisiasi mediapipe pose
pose = mpPose.Pose()
cap = cv2.VideoCapture(0) #vidio webcam

while True:
    success, img = cap.read() #pembacaan image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #koversi warna dari bgr ke rgb
    hasil = pose.process(imgRGB) #ekstraksi dari image

    if hasil.pose_landmarks:
        print("Terdeteksi")
    else:
        print("Tidak Terdeteksi")

    cv2.imshow("webcam",img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #Tutup webcam dan jendela tampilan saat "q" ditekan
cv2.destroyAllWindows()

