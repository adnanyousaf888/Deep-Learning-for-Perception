import cv2

#capturing image from camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam - Press 'q' to capture", frame)
    # USe q to capture image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('captured_image.jpg', frame)
        break
#Destroying all windows after capturing
cap.release()
cv2.destroyAllWindows()


