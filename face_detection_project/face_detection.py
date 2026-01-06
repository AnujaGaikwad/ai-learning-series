import cv2

# Use OpenCV's built-in haarcascade path
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

haar_cascade = cv2.CascadeClassifier(alg)

if haar_cascade.empty():
    raise IOError("Haar Cascade XML file not loaded")

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(10) == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
