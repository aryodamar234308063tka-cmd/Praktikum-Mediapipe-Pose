import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read() # Pembacaan image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Konversi warna dari BGR ke RGB
    hasil = pose.process(imgRGB) # Ekstraksi dari image
    if hasil.pose_landmarks:
        mpDraw.draw_landmarks(img, hasil.pose_landmarks, mpPose.POSE_CONNECTIONS) # Menggambar koneksi landmark
        for id, lm in enumerate(hasil.pose_landmarks.landmark):
            print(id, lm.x, lm.y) # Ekstraksi id, posisi x, posisi y

        cv2.imshow("webcam",img)
        cv2.waitKey(10)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()