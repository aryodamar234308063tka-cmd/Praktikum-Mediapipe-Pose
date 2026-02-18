import cv2
import mediapipe as mp
import numpy as np # Perhitungan matematika dan Operasi Vektor

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def hitung_sudut (a,b,c): # a=shoulder (pundak), b=elbow (siku), c=wrist (pergelangan tangan)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b #Vektor dari elbow ke shoulder
    bc = c - b #Vektor dari elbow ke wrist

    cosine = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) #nilai cosinus sudut siku
    sudut = np.degrees(np.arccos(cosine))

    return sudut

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hasil = pose.process(imgRGB)

    if hasil.pose_landmarks:
        mpDraw.draw_landmarks(img, hasil.pose_landmarks, mpPose.POSE_CONNECTIONS)

        lm = hasil.pose_landmarks.landmark

        # TANGAN KANAN
        shoulder_r = [lm[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_r = [lm[mpPose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mpPose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_r = [lm[mpPose.PoseLandmark.RIGHT_WRIST.value].x, lm[mpPose.PoseLandmark.RIGHT_WRIST.value].y]
        sudut_r = hitung_sudut(shoulder_r, elbow_r, wrist_r)
        # TANGAN KIRI
        shoulder_l = [lm[mpPose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_l = [lm[mpPose.PoseLandmark.LEFT_ELBOW.value].x, lm[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_l = [lm[mpPose.PoseLandmark.LEFT_WRIST.value].x, lm[mpPose.PoseLandmark.LEFT_WRIST.value].y]
        sudut_l = hitung_sudut(shoulder_l, elbow_l, wrist_l)

        TANGAN_KANAN = wrist_r[1] < shoulder_r[1] and sudut_r > 150
        TANGAN_KIRI = wrist_l[1] < shoulder_l[1] and sudut_l > 150

        if TANGAN_KANAN and TANGAN_KIRI:
            cv2.putText(img, "KEDUA TANGAN TERANGKAT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        elif TANGAN_KANAN:
            cv2.putText(img, "TANGAN KANAN TERANGKAT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        elif TANGAN_KIRI:
            cv2.putText(img, "TANGAN KIRI TERANGKAT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        else:
            cv2.putText(img, "TIDAK TERANGKAT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Deteksi Angkat Tangan",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
