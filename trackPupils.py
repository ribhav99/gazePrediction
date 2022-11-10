import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

pupil_detector = cv2.SimpleBlobDetector_create()


def detectPupils(preprocessed_frame_full, face, eye, detector):
    (x, y, width, height) = face
    (ex, ey, ew, eh) = eye

    face_frame = preprocessed_frame_full[y: y + height, x: x + width]
    eye_frame = face_frame[ey: ey + eh, ex: ex + ew]

    preprocesses_eye_frame = preprocess_eye_frame(eye_frame)

    keypoints = detector.detect(preprocesses_eye_frame)

    for keypoint in keypoints:
        x = int(keypoint.pt[0])
        y = int(keypoint.pt[1])

        eye_frame = cv2.drawMarker(
           eye_frame,
           (x, y),
           (255, 255, 0),
           markerSize = 10
           )

    cv2.imshow('preprocessed eyes', preprocesses_eye_frame)
    cv2.imshow('detection eyes', eye_frame)

    return preprocessed_frame_full


def preprocess_eye_frame(eye_frame):
    max_output_value = 100
    neighorhood_size = 99
    subtract_from_mean = 8

    eye_frame = cv2.adaptiveThreshold(
        eye_frame,
        max_output_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighorhood_size,
        subtract_from_mean
        )

    return eye_frame


def detectEyes(color_frame_full, preprocessed_frame_full, face, pupil_detector):
    (x, y, width, height) = face

    face_frame = preprocessed_frame_full[y: y + height, x: x + width]

    right_eyes = right_eye_cascade.detectMultiScale(face_frame, 1.3, 12)

    for eye in right_eyes:
        (ex, ey, ew, eh) = eye

        cv2.rectangle(
            face_frame,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
            )

        cv2.putText(
            face_frame,
            "Eye",
            (ex, ey - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

        preprocessed_frame_full = detectPupils(preprocessed_frame_full, face, eye, pupil_detector)

    return preprocessed_frame_full


def detectFaces(color_frame_full, preprocessed_frame_full, pupil_detector):
    faces = face_cascade.detectMultiScale(preprocessed_frame_full, 1.3, 5)

    for face in faces:
        (x, y, width, height) = face
        cv2.rectangle(
            preprocessed_frame_full,
            (x, y),
            (x + width, y + height),
            (255, 0, 0),
            2
            )

        cv2.putText(
            preprocessed_frame_full,
            "Face",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
            )

        preprocessed_frame_full = detectEyes(color_frame_full, preprocessed_frame_full, face, pupil_detector)

    return preprocessed_frame_full

def main():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 30
    params.filterByArea = True
    params.minArea = 35
    params.minArea = 100
    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    pupil_detector = cv2.SimpleBlobDetector_create(params)
    video_capture = cv2.VideoCapture(0)

    while True:
        flag, color_frame_full = video_capture.read()
        if not flag:
            break
        preprocessed_frame_full = cv2.cvtColor(color_frame_full, cv2.COLOR_BGR2GRAY)
        canvas = detectFaces(color_frame_full, preprocessed_frame_full, pupil_detector)
        cv2.imshow('Main Frame', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()