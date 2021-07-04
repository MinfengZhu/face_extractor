import os
import numpy as np
import scipy
import scipy.ndimage
import PIL.Image
import imageio
import matplotlib.pyplot as plt
import cv2
import face_alignment

class FaceExtrator:
    def __init__(self, output_size=1024, transform_size=4096, device='cpu', face_detector='sfd'):
        self.output_size = output_size
        self.transform_size = transform_size
        self.device = device
        self.face_detector = face_detector
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self.device, face_detector=self.face_detector)

    def get_faces(self, file_path):
        if file_path.endswith('.jpg') or file_path.endswith('.png'):
            image = np.array(PIL.Image.open(file_path))
            result = self.__extract_faces_from_image(image, vis=True)
            for face_idx in range(result['n_faces']):
                PIL.Image.fromarray(result['faces'][face_idx]).save(f'{file_path[:-4]}_{face_idx}.png')
            if 'vis' in result.keys():
                PIL.Image.fromarray(result['vis']).save(f'{file_path[:-4]}_vis.png')
        if file_path.endswith('.mp4'):
            reader = imageio.get_reader(file_path)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(f'{file_path[:-4]}_vis.mp4', fps=5)
            for frame_idx, image in enumerate(reader):
                if frame_idx % 10 != 0:
                    continue
                result = self.__extract_faces_from_image(image, vis=True)
                # for face_idx in range(result['n_faces']):
                #     PIL.Image.fromarray(result['faces'][face_idx]).save(f'{file_path[:-4]}_{frame_idx}_{face_idx}.png')
                if 'vis' in result.keys():
                    PIL.Image.fromarray(result['vis']).save(f'{file_path[:-4]}_{frame_idx}_vis.png')
                    writer.append_data(result['vis'])
            writer.close()

    def __extract_faces_from_image(self, image, vis=False):
        result = {}
        preds = self.fa.get_landmarks(image, return_bboxes=True)
        if preds == None:
            result['n_faces'] = 0 
            if vis==True: 
                result['vis'] = np.copy(image)
        else:
            landmarks, bboxes = preds
            result['landmarks'] = landmarks
            result['bboxes'] = bboxes
            result['n_faces'] = len(landmarks)
            # import pdb; pdb.set_trace()
            result['faces'] = []
            if vis==True: 
                result['vis'] = FaceExtrator.vis(np.copy(image), landmarks, bboxes)
            for idx in range(result['n_faces']):
                result['faces'].append(self.__extract_face(np.copy(image), landmarks[idx], bboxes[idx]))
        return result

    def __extract_face(self, image, landmarks, bbox):
        image = PIL.Image.fromarray(image)
        lm = landmarks
        enable_padding=True

        # Face Landmarks
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, image.size[0]), min(crop[3] + border, image.size[1]))
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - image.size[0] + border, 0), max(pad[3] - image.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image = np.pad(np.float32(image), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = image.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            image += (scipy.ndimage.gaussian_filter(image, [blur, blur, 0]) - image) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            image += (np.median(image, axis=(0,1)) - image) * np.clip(mask, 0.0, 1.0)
            image = PIL.Image.fromarray(np.uint8(np.clip(np.rint(image), 0, 255)), 'RGB')
            quad += pad[:2]
    
        # Transform.
        image = image.transform((self.transform_size, self.transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if self.output_size < self.transform_size:
            image = image.resize((self.output_size, self.output_size), PIL.Image.ANTIALIAS)
        
        return np.array(image)

    @staticmethod
    def vis(image, landmarks, bboxes):
        for bbox in bboxes:
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255,0,0), thickness=2)
        for face_landmarks in landmarks:
            for landmark in face_landmarks:
                cv2.circle(image, (int(landmark[0]), int(landmark[1])), radius=1, color=(255,0,0), thickness=2)
        return image

if __name__ == "__main__":
    fe = FaceExtrator()
    fe.get_faces('./results/girls.jpg')