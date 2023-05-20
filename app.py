from ultralytics import YOLO
import cv2

model = YOLO("./yolo.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    annotated_frame = results[0].plot()

    results = model.predict(frame, stream=True)  
    for result in results:  
        boxes = result.boxes.cpu().numpy()  
        for box in boxes: 
            r = box.xyxy[0].astype(int)
            print(r)  
            cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)  
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                result.names[int(box.cls[0])],
                (r[0] + 6, r[1] - 20),
                font,
                1.0,
                (255, 255, 255),
                1,
            )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
