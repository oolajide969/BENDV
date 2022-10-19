from concurrent.futures import thread

import pyautogui as pyautogui
from django.contrib.auth.models import User
from django.shortcuts import render
import time
from django.utils.encoding import force_str
import pyautogui
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponse
from pprint import pprint
from django.contrib.sites.shortcuts import get_current_site
from django.shortcuts import render, redirect
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.template.loader import render_to_string
from .tokens import account_activation_token
from .forms import SignUpForm, LoginForm
import mediapipe as mp
import numpy as np
import cv2
import threading
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login as dj_login
from django.contrib.auth import logout

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
poseType = ""


def detectPose(images, pose, display=True):
    # To check for Image use or Video Use
    if not display:

        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

        # Draw pose landmarks on the image.
        annotated_image = images.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        point = []
        try:
            # Extract landmarks
            point = results.pose_landmarks.landmark
        except:
            print("Halted!")
            pass

        # Return the output image and the found landmarks.
        return annotated_image, point

    else:
        # Convert the image from BGR into RGB format.
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        images.flags.writeable = False
        height, width, channels = images.shape

        # Perform the Pose Detection.
        results = pose.process(images)

        # Recoloring image from RGB image to BGR.
        images.flags.writeable = True
        images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        img = images.fromarray(images, 'RGB')
        try:
            point = results.pose_landmarks.landmark

            # Check if any landmarks are detected.
            if point:
                # Draw Pose landmarks on the output image.
                mp_drawing.draw_landmarks(
                    images, point, mp_pose.POSE_CONNECTIONS)

                # Plot the Pose landmarks in 3D.
                mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        except:
            print("Halt!")
            pass


def edshotit():
    screenshot = pyautogui.screenshot()
    screenshot.save(r'./{start_tm}_screenshot.png')


def classifyPose(landmarks, pose, output_images, display=False):
    poseType = pose
    print(poseType)
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'
    aidLabel = '..........'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)

    # Storing points into variables
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]

    # calculating the angles on each point
    leftWristAngle = calculate_angle(left_elbow, left_wrist, left_index)
    leftElbowAngle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    leftShoulderAngle = calculate_angle(left_hip, left_shoulder, left_elbow)
    leftHipAngle = calculate_angle(left_knee, left_hip, left_shoulder)
    leftKneeAngle = calculate_angle(left_ankle, left_knee, left_hip)
    leftAnkleAngle = calculate_angle(left_foot_index, left_knee, left_hip)
    rightWristAngle = calculate_angle(right_elbow, right_wrist, right_index)
    rightElbowAngle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    rightShoulderAngle = calculate_angle(right_hip, right_shoulder, right_elbow)
    rightHipAngle = calculate_angle(right_knee, right_hip, right_shoulder)
    rightKneeAngle = calculate_angle(right_ankle, right_knee, right_hip)
    rightAnkleAngle = calculate_angle(right_foot_index, right_knee, right_hip)

    angles = [leftWristAngle, leftElbowAngle, leftShoulderAngle, leftHipAngle, leftKneeAngle, leftAnkleAngle,
              rightWristAngle, rightElbowAngle, rightShoulderAngle, rightHipAngle, rightKneeAngle,
              rightAnkleAngle]

    angleStrings = ["leftWristAngle", "leftElbowAngle", "leftShoulderAngle", "leftHipAngle", "leftKneeAngle",
                    "leftAnkleAngle", "rightWristAngle", "rightElbowAngle", "rightShoulderAngle",
                    "rightHipAngle", "rightKneeAngle", "rightAnkleAngle"]

    bodyAngles = dict(zip(angleStrings, angles))
    pprint(bodyAngles, width=1)

    # Check if it is the warrior II pose or the T pose.
    # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    # Check if the both arms are straight.
    if (150 < leftElbowAngle < 180) and (150 < rightElbowAngle < 180):
        # Check if shoulders are at the required angle.
        if (70 < leftShoulderAngle < 140) and (70 < rightShoulderAngle < 140):
            # Check if it is the warrior II pose.
            # Check if one leg is straight.
            if (160 < leftKneeAngle < 180) or (160 < rightKneeAngle < 180):
                # Check if the other leg is bent at the required angle.
                if (60 < leftKneeAngle < 130) or (60 < rightKneeAngle < 130):
                    # Adjustments
                    # Specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'
                    aidLabel = 'Good  Job!'
                    # if aidLabel == 'Good Job!':
                    # screenshot = pyautogui.screenshot()sss
                    # screenshot.save(r'./screenshot.png')
                    # print('saved')
            # Check if it is the T pose.
            # Check if both legs are straight
            if (160 < leftKneeAngle < 180) and (160 < rightKneeAngle < 180):
                # Specify the label of the pose that is tree pose.
                label = 'T Pose'
                aidLabel = 'Good Job!'
                # while aidLabel == 'Good Job!':
                # t1 = threading.Thread(target=edshotit)
                # t1.start()
                # print('saved')
    # Check if it is the tree pose
    # Check if one of the legs are straight
    if (150 < leftKneeAngle <= 180) or (150 < rightKneeAngle <= 180):
        # Check if the other leg is bent at the required angle.
        if (25 < leftKneeAngle < 80) or (25 < rightKneeAngle < 80):
            label = 'Tree Pose'
            aidLabel = 'Good Job!'
            # while aidLabel == 'Good Job!':
            # thread.start_new_thread(edshotit())
            # print('saved')
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        # Update the color (to green) with which the label will be written on the image.b
        color = (0, 255, 0)

    # Aid Label Feed
    # T Pose
    if poseType == 'T Pose':
        if (leftElbowAngle < 150) or (rightElbowAngle < 150):
            aidLabel = 'Straighten Arm'
        if (leftShoulderAngle < 70) or (rightShoulderAngle < 70):
            aidLabel = 'Raise your Arm'
        if (leftShoulderAngle > 150) or (rightShoulderAngle > 150):
            aidLabel = 'Arms are too high'
        if (leftKneeAngle < 160) or (rightKneeAngle < 160):
            aidLabel = 'Your Legs are not Straight'
    if poseType == 'Warrior II Pose':
        if (leftElbowAngle < 150) or (rightElbowAngle < 150):
            aidLabel = 'Straighten Arm'
        if (leftShoulderAngle < 70) or (rightShoulderAngle < 70):
            aidLabel = 'Raise your Arm'
        if (leftShoulderAngle > 150) or (rightShoulderAngle > 150):
            aidLabel = 'Arms are too high'
        if (leftKneeAngle < 60) or (rightKneeAngle < 60):
            aidLabel = 'That knee is overly bent'
        if (leftKneeAngle > 130) and (rightKneeAngle > 130):
            aidLabel = 'Warrior Pose not Break Dance'
    if poseType == 'Tree Pose':
        if (leftKneeAngle > 150) and (rightKneeAngle > 150):
            aidLabel = "You're standing!"
        if (80 < leftKneeAngle < 150) or (80 < rightKneeAngle < 150):
            aidLabel = "Bend your knee a little more"
    # Write the label on the output image.
    cv2.putText(output_images, label, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, color, 5)
    cv2.putText(output_images, aidLabel, (1300, 1050), cv2.FONT_HERSHEY_PLAIN, 3, color, 5)

    ##while True:

    if display:
        # Write the label on the output image.
        cv2.putText(output_images, label, (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, color, 5)
        cv2.putText(output_images, aidLabel, (1300, 1050), cv2.FONT_HERSHEY_PLAIN, 3, color, 5)

    else:
        # Return the output image and the classified label.
        return output_images, label, bodyAngles


def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    radians = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] -
                                                                                    point2[0])
    angle_here = np.abs(radians * 180.0 / np.pi)

    if angle_here > 180.0:
        angle_here = 360 - angle_here

    return angle_here


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def drop_feed(self):
        self.video.release()
        self.video.destroyAllWindows()

    def get_frame(self, pose):
        image = self.frame
        frame, landmarks = detectPose(image, pose_video, display=False)
        if landmarks:
            frame, label, bodyAngles = classifyPose(landmarks, pose, frame, display=False)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera, pose):
    while True:
        frame = camera.get_frame(pose)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
@login_required
def livefeed(request):
    pose = request.session['pose']
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam, pose), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass


def index(request):
    return render(request, "index.html")


@login_required
def vid(request, pose):
    request.session['pose'] = pose
    return render(request, "video.html")


def login(request):
    form = LoginForm()
    message = ''
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data['username'],
                password=form.cleaned_data['password'],
            )
            dj_login(request, user)
            if user is not None:
                return redirect('/projects/home/')
            else:
                return HttpResponse("Invalid login details")
    return render(request, "login.html", {'form': form})


def logout_view(request):
    VideoCamera.drop_feed()
    logout(request)
    return redirect('/projects/index/')


def about(request):
    return render(request, "about.html")


def schedule(request):
    return render(request, "schedule.html")


def sessions(request):
    context = {'Tree': 'Tree Pose',
               'Warrior': 'Warrior II Pose',
               'Tpose': 'T Pose'}
    return render(request, "sessions.html", context)


def home(request):
    context = {'Tree': 'Tree Pose',
               'Warrior': 'Warrior II Pose',
               'Tpose': 'T Pose'}
    return render(request, "home.html", context)


def blog(request):
    return render(request, "blog.html")


def singleBlog(request):
    return render(request, "blog-single.html")


def contact(request):
    return render(request, "contact.html")


def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            current_site = get_current_site(request)
            subject = 'Activate Your Yoga Account'
            message = render_to_string('account_activation_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                'token': account_activation_token.make_token(user),
            })
            user.email_user(subject, message)
            return redirect('/projects/account_activation_sent/')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


def account_activation_sent(request):
    return render(request, 'account_activation_sent.html')


def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, user.DoesNotExist):
        user = None

    if user is not None and account_activation_token.check_token(user, token):
        user.is_active = True
        user.profile.email_confirmed = True
        user.save()
        dj_login(request, user)
        return redirect('/projects/login/')
    else:
        return render(request, 'account_activation_invalid.html')
