import cv2
import mediapipe as mp
import numpy as np

np.set_printoptions(threshold=np.inf)
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec
mp_face_mesh = mp.solutions.face_mesh
# 图片人物抠图:
IMAGE_FILES = []
BG_COLOR = (255, 255, 255)
MASK_COLOR = (255, 255, 255)
file = '1.png'
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    fg_image = np.ones(image.shape, dtype=np.uint8)
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.waitKey(0)
    cv2.imwrite('selfie1.png', output_image)

# 静态图片:
IMAGE_FILES = ["1.png"]
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        a = np.zeros((256, 256, 3))
        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = a.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS, )
        print(annotated_image.shape)
        print(annotated_image[:, 156, 1], annotated_image[:, 156, 2])
        b = np.zeros((256, 256, 3))
        cv2.imwrite('a' + str(idx) + '.png', b)
        cv2.imwrite('a1' + str(idx) + '.png', annotated_image)
        b[:, :, 0] = annotated_image[:, :, 1]
        cv2.imwrite('a2' + str(idx) + '.png', b)
        m = annotated_image[:, :, :1]
        print(m.shape)
        print((m == b[:, :, 0]).all())

