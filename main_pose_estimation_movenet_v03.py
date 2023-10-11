import numpy as np
from collections import namedtuple
import cv2
from pathlib import Path
import math
from FPS import FPS, now
import argparse
import os
import json
from collections import OrderedDict
from openvino.inference_engine import IENetwork, IECore
from Tracker import TrackerIoU, TrackerOKS, TRACK_COLORS
import streamlit as st
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
#import easygui


SIDE = 1
FRONT = 0
BACK = 2

max_img_size = 1024

photo_mode = SIDE
file_data = OrderedDict()
keypoints_list = np.zeros([16, 2])
index = 0
count = 0
center_vector=[0.0, 0.0]
shoulder_vector =[0.0, 0.0]
hip_vector = [0.0, 0.0]
#shoulder_angle =0.0
#hip_angle = 0.0
#neck_angle =0.0

shoulder_r = [0.0, 0.0]
shoulder_l = [0.0, 0.0]
hip_r = [0.0, 0.0]
hip_l = [0.0, 0.0]
knee_r = [0.0, 0.0]
knee_l = [0.0, 0.0]
#center_up =[0.0, 0.0]
#center_down = [0.0, 0.0]
#shoulder_mean =[0.0, 0.0]
#hip_mean =[0.0, 0.0]
#ear_mean =[0.0, 0.0] 

global frame_front
global frame_back
global frame_side

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models/movenet_multipose_lightning_256x256_FP32.xml"
            
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# LINES_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]

class Body:
    def __init__(self, score, xmin, ymin, xmax, ymax, keypoints_score, keypoints, keypoints_norm):
        self.score = score # global score
        # xmin, ymin, xmax, ymax : bounding box
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoints_score = keypoints_score# scores of the keypoints
        self.keypoints_norm = keypoints_norm # keypoints normalized ([0,1]) coordinates (x,y) in the input image
        self.keypoints = keypoints # keypoints coordinates (x,y) in pixels in the input image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

    def str_bbox(self):
        return f"xmin={self.xmin} xmax={self.xmax} ymin={self.ymin} ymax={self.ymax}"

# Padding (all values are in pixel) :
# w (resp. h): horizontal (resp. vertical) padding on the source image to make its ratio same as Movenet model input. 
#               The padding is done on one side (bottom or right) of the image.
# padded_w (resp. padded_h): width (resp. height) of the image after padding
Padding = namedtuple('Padding', ['w', 'h', 'padded_w',  'padded_h'])

def front_pose_estimator(key):
    shoulder_r = key[5]
    shoulder_l = key[6]
    hip_r = key[11]
    hip_l = key[12]
    knee_r = key[13]
    knee_l = key[14]
    center_up =(shoulder_r+ shoulder_l)/2
    center_down = (hip_r+ hip_l)/2 
    center_vector = center_up- center_down
    shoulder_vector = shoulder_r- center_up
    hip_vector = hip_r - center_down
    shoulder_angle =270 - 180*skelecton_angle(center_down, center_up, shoulder_r)/np.pi 
    hip_angle = 90- 180*skelecton_angle(center_up, center_down, hip_r)/np.pi
    return shoulder_angle, hip_angle, center_up, center_down

def back_pose_estimator(key):
    shoulder_r = key[5]
    shoulder_l = key[6]
    hip_r = key[11]
    hip_l = key[12]
    knee_r = key[13]
    knee_l = key[14]
    center_up =(shoulder_r+ shoulder_l)/2
    center_down = (hip_r+ hip_l)/2 
    center_vector = center_up- center_down
    shoulder_vector = shoulder_r- center_up
    hip_vector = hip_r - center_down
    shoulder_angle = 90 - 180*skelecton_angle(center_down, center_up, shoulder_r)/np.pi 
    hip_angle = 270 - 180*skelecton_angle(center_up, center_down, hip_r)/np.pi
    return shoulder_angle, hip_angle, center_up, center_down

def side_pose_estimator(key):
    shoulder_mean = key[5] #(key[5] + key[6])/2
    hip_mean = key[11] #(key[11]+ key[12])/2
    ear_mean =key[3] #(key[3] + key[4])/2
    neck_angle =180- 180*skelecton_angle(hip_mean, shoulder_mean, ear_mean)/np.pi
    return shoulder_mean, hip_mean, ear_mean, neck_angle 
    
def skelecton_angle(p1, p2, p3):
    # p1 base keypoint, p2 joint keypoint, p3 end keypoint
    alpha = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    beta = np.array([p3[0]-p2[0], p3[1]-p2[1]])

    theta = np.arccos(np.inner(alpha, beta)/(np.linalg.norm(alpha, ord=2)*np.linalg.norm(beta, ord=2)))
    normal_vector = np.cross(alpha, beta)

    if normal_vector > 0 :
        theta = theta
    else:
        theta = 2* np.pi - theta 
    return theta    
def report_img(img):
    img_w_max = 160
    img_h, img_w = img.shape[:2]
    scale = 1
    if img_w > img_w_max:
        scale = img_w_max /img_w
    print(img_h, img_w)
    img_h_r = (int)(img_h * scale)
    print(320, img_h_r)
    img_resize = cv2.resize(img, (img_w_max, img_h_r))
    #return Image.fromarray(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)) 
    return Image.fromarray(img_resize)

def report_gen(f_img, s_img, b_img):
    c=canvas.Canvas('report.pdf', pagesize= A4)
    c.drawString(100, 750, "Posture Estimation Result")
    
    front_img = report_img(f_img)
    side_img = report_img(s_img)
    back_img = report_img(b_img)
#    front_img = report_img(frame_front)
#    side_img = report_img(frame_side)
#    back_img = report_img(frame_back)

    c.drawInlineImage(front_img, 50, 500, width =None, height=None )
    c.drawString(250, 700, "Front Pose")
    c.drawInlineImage(side_img, 50, 275, width =None, height=None )
    c.drawString(250, 475, "Side Pose")
    c.drawInlineImage(back_img, 50, 50, width =None, height=None )
    c.drawString(250, 250, "Back Pose")
    c.showPage()
#    file_absolut_path = easygui.fileopenbox(title='Add File', default="*.*")
#    st.write(file_absolut_path)
    c.save()

class MovenetMPOpenvino:
    def __init__(self, input_src=None,
                xml=DEFAULT_MODEL, 
                device="CPU", #CPU default
                tracking=False,
                score_thresh=0.2,
                output=None):
        
        self.score_thresh = score_thresh
        self.tracking = tracking

        if tracking is None:
            self.tracking = False

        #if input_src.endswith('.jpg') or input_src.endswith('.png') :
        self.input_type= "image"
       # encoding_img = np.fromstring(img_data, dtype = np.uint8)
       # self.img = cv2.imdecode(encoding_img, cv2.IMREAD_COLOR)
       # self.img = cv2.imread(input_src)
        pil_img = Image.open(img_data)
        self.img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.video_fps = 25

        self.img_h, self.img_w = self.img.shape[:2]
        if self.img_h < self.img_w:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        self.img_h, self.img_w = self.img.shape[:2]
        print(self.img_h, self.img_w)
        if self.img_h >= max_img_size:
                scale = max_img_size/self.img_h
                self.img = cv2.resize(self.img, ((int)(self.img_w*scale), (int)(self.img_h*scale)))
        self.img_h, self.img_w = self.img.shape[:2]
  
        # Load Openvino models
        self.load_model(xml, device)     

        # Rendering flags
        self.show_fps = True
        self.show_bounding_box = False

        if output is None: 
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(self.img_w, self.img_h)) 

        # Define the padding
        # Note we don't center the source image. The padding is applied
        # on the bottom or right side. That simplifies a bit the calculation
        # when depadding
        if self.img_w / self.img_h > self.pd_w / self.pd_h:
            pad_h = int(self.img_w * self.pd_h / self.pd_w - self.img_h)
            self.padding = Padding(0, pad_h, self.img_w, self.img_h + pad_h)
        else:
            pad_w = int(self.img_h * self.pd_w / self.pd_h - self.img_w)
            self.padding = Padding(pad_w, 0, self.img_w + pad_w, self.img_h)
        print("Padding:", self.padding)
        
    def load_model(self, xml_path, device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(device)
        print("{}{}".format(" "*8, device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[device].major, versions[device].minor))
        print("{}Build ........... {}".format(" "*8, versions[device].build_number))

        name = os.path.splitext(xml_path)[0]
        bin_path = name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(xml_path, bin_path))
        self.pd_net = self.ie.read_network(model=xml_path, weights=bin_path)
        # Input blob: input:0 - shape: [1, 3, 256, 256] (lightning)
        # Output blob: Identity - shape: [1, 6, 56]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}")
        _,_,self.pd_h,self.pd_w = self.pd_net.input_info[self.pd_input_blob].input_data.shape
        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
        self.pd_kps = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        self.infer_nb = 0
        self.infer_time_cumul = 0

    def pad_and_resize(self, frame):
        """ Pad and resize the image to prepare for the model input."""

        padded = cv2.copyMakeBorder(frame, 
                                        0, 
                                        self.padding.h,
                                        0, 
                                        self.padding.w,
                                        cv2.BORDER_CONSTANT)

        padded = cv2.resize(padded, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)

        return padded

    def pd_postprocess(self, inference):
        result = np.squeeze(inference[self.pd_kps]) # 6x56
        bodies = []
        for i in range(6):
            kps = result[i][:51].reshape(17,-1)
            bbox = result[i][51:55].reshape(2,2)          
            score = result[i][55]
            if score > self.score_thresh:
                ymin, xmin, ymax, xmax = (bbox * [self.padding.padded_h, self.padding.padded_w]).flatten().astype(int)
                kp_xy =kps[:,[1,0]]
                keypoints = kp_xy * np.array([self.padding.padded_w, self.padding.padded_h])

                body = Body(score=score, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, 
                            keypoints_score = kps[:,2], 
                            keypoints = keypoints.astype(int),
                            keypoints_norm = keypoints / np.array([self.img_w, self.img_h]))
                bodies.append(body)
        return bodies
        

    def pd_render(self, frame, bodies):
        global file_data
        global index
        global count
        global keypoints_list
        
        thickness = 3 
        color_skeleton = (255, 180, 90)
        color_box = (0,255,255)
        for body in bodies:
            if self.tracking:
                color_skeleton = color_box = TRACK_COLORS[body.track_id % len(TRACK_COLORS)]
                
            lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.keypoints_score[line[0]] > self.score_thresh and body.keypoints_score[line[1]] > self.score_thresh]
            cv2.polylines(frame, lines, False, color_skeleton, 2, cv2.LINE_AA)
 
           # print(body.keypoints)
            """for i in range(len(lines)) :
            	lines[i] = lines[i].tolist()
            
            file_data[index] = [body.keypoints.tolist(), lines]
            index += 1"""
            
            keypoints_list = body.keypoints

            flag = 0
            
            for i,x_y in enumerate(body.keypoints):
               # if math.dist(ref_data[str(count)][0][i], x_y) < 50 :
               #     color = (0,255,0)
               # else :
               #     color = (0,0,255)
               #     flag = 1
                color=(0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)
            
            if flag == 0 :
            	count += 1

            if self.show_bounding_box:
                cv2.rectangle(frame, (body.xmin, body.ymin), (body.xmax, body.ymax), color_box, thickness)

            if self.tracking:
                # Display track_id at the center of the bounding box
                x = (body.xmin + body.xmax) // 2
                y = (body.ymin + body.ymax) // 2
                cv2.putText(frame, str(body.track_id), (x,y), cv2.FONT_HERSHEY_PLAIN, 4, color_box, 3)
                
    def run(self):

        self.fps = FPS()
        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0
        global hip_mean
        global shoulder_mean
        global ear_mean

        first_run =1

        while True:
          #  self.img = cv2.imread(self.input_src)

            frame = self.img.copy()
            padded = self.pad_and_resize(frame)
                     
            frame_nn = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)[None,] 
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            bodies = self.pd_postprocess(inference)
            if self.tracking:
                bodies = self.tracker.apply(bodies, now())
            self.pd_render(frame, bodies)
            nb_pd_inferences += 1

            self.fps.update()               

            if self.show_fps:
                self.fps.draw(frame, orig=(50,50), size=1, color=(240,180,100))
          
            if photo_mode == FRONT:
                shoulder_angle, hip_angle, center_up, center_down = front_pose_estimator(keypoints_list)
                center_up_x = (int)(center_up[0])
                center_up_y = (int)(center_up[1])
                center_down_x = (int)(center_down[0])
                center_down_y = (int)(center_down[1])
                x_offset = np.sqrt(np.square(center_up_x - keypoints_list[5][0])+np.square(center_up_y - keypoints_list[5][1]))
                if (center_up_y - center_down_y) != 0:
                    base_line_gradient = (-1) * (center_up_x - center_down_x)/(center_up_y - center_down_y)
                else:
                    base_line_gradient = 0
                extLine_up_x = (int)(center_up_x + x_offset)
                extLine_up_y = (int)(base_line_gradient * x_offset + center_up_y )

                extLine_down_x = (int)(center_down_x + x_offset)
                extLine_down_y = (int)(base_line_gradient * x_offset + center_down_y )
                cv2.line(frame, (center_up_x, center_up_y), (extLine_up_x, extLine_up_y), (255,0,0),2)
                cv2.line(frame, (center_down_x, center_down_y), (extLine_down_x, extLine_down_y),(255,0,0),2)
                cv2.line(frame, (center_down_x, center_down_y), (center_up_x, center_up_y),(255,0,0),2)
                cv2.putText(frame, "shoulder angle %f" %shoulder_angle, (extLine_up_x, extLine_up_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)                
                cv2.putText(frame, "hip angle %f" %hip_angle, (extLine_down_x, extLine_down_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)                
 
            if photo_mode == SIDE:
                shoulder_mean, hip_mean, ear_mean, neck_angle = side_pose_estimator(keypoints_list)
                hip_mean_x = (int)(hip_mean[0])
                hip_mean_y = (int)(hip_mean[1])
                shoulder_mean_x = (int)(shoulder_mean[0])
                shoulder_mean_y = (int)(shoulder_mean[1])
                if (shoulder_mean_x - hip_mean_x) != 0:
                    base_line_gradient = (shoulder_mean_y - hip_mean_y)/(shoulder_mean_x-hip_mean_x)
                else:
                    base_line_gradient = 0
                x_offset = 10
                extLine_x = (int)(shoulder_mean_x + x_offset)
                extLine_y = (int)(base_line_gradient * x_offset + shoulder_mean_y)
                ear_mean_x = (int)(ear_mean[0])
                ear_mean_y = (int)(ear_mean[1])
                
                #print((int)(hip_mean[0]), (int)(hip_mean[1]))
                #print(shoulder_mean)
                cv2.line(frame, (hip_mean_x, hip_mean_y), (extLine_x, extLine_y), (255,0,0),2)
                cv2.line(frame, (shoulder_mean_x, shoulder_mean_y), (ear_mean_x, ear_mean_y),(255,0,0),2)
                cv2.putText(frame, "neck angle %f" %neck_angle, (ear_mean_x, ear_mean_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)                
            # print(shoulder_angle[file_index], hip_angle[file_index])

            if photo_mode == BACK :
                shoulder_angle, hip_angle, center_up, center_down = back_pose_estimator(keypoints_list)
                center_up_x = (int)(center_up[0])
                center_up_y = (int)(center_up[1])
                center_down_x = (int)(center_down[0])
                center_down_y = (int)(center_down[1])
                x_offset = np.sqrt(np.square(center_up_x - keypoints_list[5][0])+np.square(center_up_y - keypoints_list[5][1]))
                if (center_up_y - center_down_y) != 0:
                    base_line_gradient = (-1) * (center_up_x - center_down_x)/(center_up_y - center_down_y)
                else:
                    base_line_gradient = 0
                extLine_up_x = (int)(center_up_x + x_offset)
                extLine_up_y = (int)(base_line_gradient * x_offset + center_up_y )

                extLine_down_x = (int)(center_down_x + x_offset)
                extLine_down_y = (int)(base_line_gradient * x_offset + center_down_y )
                cv2.line(frame, (center_up_x, center_up_y), (extLine_up_x, extLine_up_y), (255,0,0),2)
                cv2.line(frame, (center_down_x, center_down_y), (extLine_down_x, extLine_down_y),(255,0,0),2)
                cv2.line(frame, (center_down_x, center_down_y), (center_up_x, center_up_y),(255,0,0),2)
                cv2.putText(frame, "shoulder angle %f" %shoulder_angle, (extLine_up_x, extLine_up_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)                
                cv2.putText(frame, "hip angle %f" %hip_angle, (extLine_down_x, extLine_down_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)                

            #cv2.imshow("Movenet", frame)
            if first_run == 1 :
                if photo_mode == FRONT :
                    #col1.image(frame, width = 500)
                    col1.image(frame)
                    col1.write( "shoulder angle : %f " %shoulder_angle)
                    col1.write( "hip angle      : %f " %hip_angle)
                    return frame
                    break
                elif photo_mode == SIDE :
                    #col2.image(frame, width = 500)
                    col2.image(frame)
                    col2.write("neck angle   : %f " %neck_angle)
                    return frame
                    break
                elif photo_mode == BACK :
                    col1.subheader("Back")
                    col1.image(frame)
                    col1.write( "shoulder angle : %f " %shoulder_angle)
                    col1.write( "hip angle      : %f " %hip_angle)
                    return frame
                first_run = 0
            #frame_placeholder.image(frame, channels = "BGR")
                  
            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, frame)
                else:
                    self.output.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: #or stop_button_pressed:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)

        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()

            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")

        if self.output and self.input_type != "image":
            self.output.release()
        
        with open("points_lines_data", 'w', encoding = "utf-8") as make_file :
        	json.dump(file_data, make_file, ensure_ascii = False, indent="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='4', 
                        help="Path to video or image file to use as input (default=%(default)s)")
    # parser.add_argument("-p", "--precision", type=int, choices=[16, 32], default=32,
    #                     help="Precision (default=%(default)i")                    
    parser.add_argument("--xml", type=str,
                        help="Path to an .xml file for model")
    parser.add_argument("-r", "--res", default="256x256", choices=["192x192", "192x256", "256x256", "256x320", "320x320", "480x640", "736x1280"])
    #parser.add_argument("-d", "--device", default='CPU', type=str,
                        #help="Target device to run the model (default=%(default)s)") 
    parser.add_argument("-t", "--tracking", choices=["iou", "oks"],
                        help="Enable tracking and specify method")
    parser.add_argument("-s", "--score_threshold", default=0.2, type=float,
                        help="Confidence score (default=%(default)f)")                     
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    args = parser.parse_args()

    dumb_photo = "./blank.jpg"
    frame_front = cv2.imread(dumb_photo)
    frame_side = cv2.imread(dumb_photo)
    frame_back = cv2.imread(dumb_photo)

    if not args.xml:
        args.xml = SCRIPT_DIR / f"models/movenet_multipose_lightning_{args.res}_FP32.xml"

    st.set_page_config(page_title="Pose estimation")
    st.title("Pose estimation")
    st.sidebar.write("## Upload Front photo")
 
    col1, col2 = st.columns(2)
    col1.subheader("Front Pose ")
    col2.subheader("Side pose")
    front_upload = st.sidebar.file_uploader("Upload Front Image", type=["png", "jpg","jpeg"])
    side_upload = st.sidebar.file_uploader("Upload Side Image", type=["png", 'jpg', 'jpeg'])
    back_upload = st.sidebar.file_uploader("Upload Back Image", type=["png", 'jpg', 'jpeg'])
 
    if front_upload is not None:
        print("front mode")
        print(front_upload)
        #col1.image(front_upload)
        photo_mode = FRONT
        img_data = front_upload
        print(img_data.type)
        pd = MovenetMPOpenvino(input_src = dumb_photo, ##input_src=args.input, 
                    xml=args.xml,
                    #device=args.device, 
                    tracking=args.tracking,
                    score_thresh=args.score_threshold,
                    output=args.output)
        frame_front = pd.run()
        st.write(front_upload)
    else:
        col1.image(dumb_photo)

    if side_upload is not None:
        print("side mode")
        print(side_upload)
        #col2.image(side_upload)
        photo_mode = SIDE
        img_data = side_upload
        pd = MovenetMPOpenvino(input_src = dumb_photo, ##input_src=args.input, 
                    xml=args.xml,
                    #device=args.device, 
                    tracking=args.tracking,
                    score_thresh=args.score_threshold,
                    output=args.output)
        frame_side = pd.run()
        st.write(side_upload)
    else:
        col2.image(dumb_photo)

    if back_upload is not None:
        print("back mode")
        print(side_upload)
        #col2.image(side_upload)
        photo_mode = BACK
        img_data = back_upload
        pd = MovenetMPOpenvino(input_src = dumb_photo, ##input_src=args.input, 
                    xml=args.xml,
                    #device=args.device, 
                    tracking=args.tracking,
                    score_thresh=args.score_threshold,
                    output=args.output)
        frame_back = pd.run()
        st.write(back_upload)
    else:
        col1.subheader("Back")
        col1.image(dumb_photo)

    with st.sidebar:
        if st.button("Report Save"):
            report_gen(frame_front, frame_side, frame_back)
    
    
