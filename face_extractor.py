import os
from tqdm import tqdm
import numpy as np
import scipy
import scipy.ndimage
import PIL.Image
import imageio
import matplotlib.pyplot as plt
import cv2
import face_alignment
from youtubedl import YouTubeDownloader
import motrackers

class FaceExtrator:
    def __init__(self, output_size=1024, transform_size=4096, device='cpu', face_detector='sfd'):
        self.output_size = output_size
        self.transform_size = transform_size
        self.device = device
        self.face_detector = face_detector
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self.device, face_detector=self.face_detector)
        self.tracker = motrackers.SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)

    def get_faces(self, filepath):
        if filepath.endswith('.jpg') or filepath.endswith('.png'):
            image = np.array(PIL.Image.open(filepath))
            result = self.__extract_faces_from_image(image, vis=True)
            for face_idx in range(result['n_faces']):
                PIL.Image.fromarray(result['faces'][face_idx]).save(f'{file_path[:-4]}_{face_idx}.png')
            if 'vis' in result.keys():
                PIL.Image.fromarray(result['vis']).save(f'{file_path[:-4]}_vis.png')
        elif filepath.endswith('.mp4') or filepath.endswith('.webm') or filepath.endswith('.avi'):
            ext = filepath.split('.')[-1]
            filename = filepath[:-len(ext)]
            reader = imageio.get_reader(filepath)
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(f'{filename}_vis.mp4', fps=5, quality=10, format='FFMPEG', codec='libx265', pixelformat='yuv444p')
            self.filename = filename
            self.face_writers = []
            with tqdm(total=reader.count_frames()) as pbar:
                for frame_idx, image in enumerate(reader):
                    if frame_idx < 500 or frame_idx > 1000 or frame_idx % 10 !=0:
                        pbar.update(1)
                        continue
                    result = self.__extract_faces_from_image(image, vis=True)

                    # for face_idx in range(result['n_faces']):
                        # PIL.Image.fromarray(result['faces'][face_idx]).save(f'{filename}_{frame_idx}_{face_idx}.png')
                    if 'vis' in result.keys():
                        PIL.Image.fromarray(result['vis']).save(f'{filename}_{frame_idx}_vis.png')
                        writer.append_data(result['vis'])
                    pbar.update(1)
                writer.close()
                for face_writer in self.face_writers:
                    face_writer.close()
        elif filepath.startswith('https://www.youtube.com/watch?v='):
            yt_dl = YouTubeDownloader()
            filepath = yt_dl.download(filepath)
            self.get_faces(filepath)
            #self.get_faces(f"results/{file_path.replace('https://www.youtube.com/watch?v=', '')}.mp4")
    
    def __face_detection(self, image):
        image_det = PIL.Image.fromarray(image)
        det_scale = 1
        while image.shape[1] * 1.0 / det_scale > 1500:
            det_scale *= 2
        image_det = image_det.resize((int(image_det.size[0] *1.0 / det_scale), int(image_det.size[1] * 1.0 / det_scale)))
        image_det = np.array(image_det)
        bboxes = self.fa.face_detector.detect_from_image(np.copy(image_det))
        if bboxes is None:
            return bboxes
        else:
            bboxes = np.stack(bboxes)
            bboxes[:,:4] = bboxes[:,:4] * det_scale
            return bboxes

    def __landmark_detection(self, image, bboxes):
        image_det = PIL.Image.fromarray(image)
        det_scale = 1
        while image.shape[1] * 1.0 / det_scale > 1500:
            det_scale *= 2
        image_det = image_det.resize((int(image_det.size[0] *1.0 / det_scale), int(image_det.size[1] * 1.0 / det_scale)))
        image_det = np.array(image_det)
        landmarks = self.fa.get_landmarks(np.copy(image_det), detected_faces=bboxes / det_scale)
        landmarks *= det_scale
        return landmarks
       
    def __extract_faces_from_image(self, image, vis=False):
        result = {}
        bboxes = self.__face_detection(np.copy(image))
        if bboxes is None:
            result['n_faces'] = 0
            if vis==True:
                result['vis'] = np.copy(image)
        else:
            detection_bboxes = bboxes[:,:4]
            detection_confidences = bboxes[:,4:]
            detection_class_ids = np.zeros_like(detection_confidences)
            output_tracks = self.tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
            refined_bboxes = []
            for track in output_tracks:
                frame, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
                assert len(track) == 10
                refined_bboxes.append([bb_left, bb_top, bb_width, bb_height])
                if id >= len(self.face_writers):
                    self.face_writers.append(imageio.get_writer(f'{self.filename}_face{id}.mp4', fps=1, quality=10, format='FFMPEG', codec='libx265', pixelformat='yuv444p'))
            refined_bboxes = np.array(refined_bboxes)  
            landmarks = self.__landmark_detection(np.copy(image), refined_bboxes)


            result['landmarks'] = landmarks
            result['bboxes'] = refined_bboxes
            result['n_faces'] = len(landmarks)
            result['faces'] = []
            if vis==True:
                result['vis'] = FaceExtrator.vis(np.copy(image), landmarks, refined_bboxes)
            # import pdb; pdb.set_trace()
            for i, track in enumerate(output_tracks):
                frame, track_id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
                face = self.__extract_face(np.copy(image), landmarks[i], refined_bboxes[i])
                result['faces'].append(face)
                self.face_writers[track_id].append_data(face)       
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
    # fe.get_faces('./results/girls.jpg')
    fe.get_faces('results/Interview with My Twin Brother.mp4')
    # fe.get_faces('https://www.youtube.com/watch?v=8ZKzx1C4-DY')

