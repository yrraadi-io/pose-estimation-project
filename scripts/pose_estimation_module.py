import cv2
import mediapipe as mp
import time
import ffmpeg


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks and not static_image_mode
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation and not static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_draw = mp.solutions.drawing_utils
        # create mp solutions pose object
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )

    def find_pose(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

        return img

    def get_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list

    # method to convert the video resolution
    def video_convert(self):
        cap = cv2.VideoCapture("./pose-videos/golf.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            "./pose-videos/golf_compressed.mp4", fourcc, 30, (568, 320)
        )

        while True:
            ret, frame = cap.read()
            if ret == True:
                b = cv2.resize(
                    frame, (568, 320), fx=0, fy=0, interpolation=cv2.INTER_CUBIC
                )
                out.write(b)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # method to fix the video orientation
    def check_rotation(self, path_video_file):
        # this returns meta-data of the video file in form of a dictionary
        meta_dict = ffmpeg.probe(path_video_file)

        # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
        # we are looking for
        rotateCode = None
        if int(meta_dict["streams"][0]["tags"]["rotate"]) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict["streams"][0]["tags"]["rotate"]) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

        return rotateCode

    def correct_rotation(self, frame, rotateCode):
        return cv2.rotate(frame, rotateCode)


def main():
    detector = PoseDetector()
    # detector.video_convert()
    cap = cv2.VideoCapture("./pose-videos/golf.mp4")
    # rotate_code = detector.check_rotation("./pose-videos/golf.mp4")
    p_time = 0

    while True:
        success, img = cap.read()
        # if rotateCode is not None:
        #     img = detector.correct_rotation(img, rotateCode)
        img = detector.find_pose(img=img)
        lm_list = detector.get_position(img)
        if len(lm_list) != 0:
            print(lm_list)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # put fps in the image
        cv2.putText(
            img,
            str(int(fps)),
            (70, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 0, 255),
            3,
        )
        # display the image
        cv2.imshow("image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
