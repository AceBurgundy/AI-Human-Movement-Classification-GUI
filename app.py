from customtkinter import CTk, CTkFrame, CTkButton, CTkLabel, CTkImage, CTkFont # type: ignore
from cv2 import VideoCapture, resize, cvtColor, COLOR_BGR2RGB # type: ignore
from tf_pose.estimator import TfPoseEstimator, Human
from tf_pose.networks import get_graph_path, model_wh
from typing import List, Dict, Optional, Tuple
from customtkinter import ThemeManager
from PIL import Image
from enum import Enum

class Movement(Enum):

    STANDING = "Standing"
    RUNNING = "Running"
    SITTING = "Sitting"
    WALKING = "Walking"
    JUMPING = "Jumping"
    ALL = "All"

def get_color(class_name: str, property: str = "fg_color") -> tuple[str, str]:
    """
    Returns the specific fg_color of the class that has been passed
    """
    return ThemeManager.theme[class_name][property]

class App(CTk):

    def __init__(self) -> None:
        """
        Initializes the app
        """
        super().__init__()
        self.title = "Human Movement Classification"

        self.app_width = 1080
        self.app_height = 670
        self.geometry(f"{self.app_width}x{self.app_height}")

        self.button_frame: CTkFrame = CTkFrame(self)
        self.button_frame.pack(side="left", pady=10, padx=10, fill="y")

        self.movements: List[Movement] = [
            Movement.STANDING,
            Movement.RUNNING,
            Movement.SITTING,
            Movement.WALKING,
            Movement.JUMPING,
            Movement.ALL
        ]

        self.preferred_movement: Movement = self.movements[0]
        self.buttons: List[CTkButton] = []  # List to store the buttons

        for index, movement in enumerate(self.movements):
            # Create buttons and store them in the list
            button: CTkButton = CTkButton(
                self.button_frame,
                text=movement.value,
                command=lambda movement=movement: self.set_new_movement(movement)
            )

            button.pack(pady=(10, 0), padx=10, side="top")
            self.buttons.append(button)

        self.update_button_colors()
        self.update_idletasks()

        video_frame_width = self.app_width - self.button_frame.winfo_width()
        self.video_frame_size = (video_frame_width, self.app_height)

        # Create a label for displaying the camera feed
        self.video_frame: CTkLabel = CTkLabel(
            self,
            width=video_frame_width,
            height=self.app_height,
            text='',
            font=CTkFont(family="Arial", size=30),
            text_color="light green",
        )

        self.video_frame.pack(padx=(0, 10), pady=10, side="right")

        self.resize: str = '0x0'  # Resize dimensions for the model input
        self.resize_out_ratio: float = 4.0  # Output resize ratio for upsampling
        self.model_name: str = 'mobilenet_thin'  # Model name for pose estimation
        self.show_process: bool = False  # Flag to show intermediate process steps
        self.use_tensorrt: bool = False  # Flag for TensorRT optimization

        # Get model input size (width, height) based on the specified resize string
        self.model_width, self.model_height = model_wh(self.resize)

        # pose estimator based on model size or default dimensions
        if self.model_width > 0 and self.model_height > 0:
            self.pose_estimator: TfPoseEstimator = TfPoseEstimator(get_graph_path(self.model_name), target_size=(self.model_width, self.model_height), trt_bool=self.use_tensorrt)
        else:
            self.pose_estimator = TfPoseEstimator(get_graph_path(self.model_name), target_size=(640, 480), trt_bool=self.use_tensorrt)

        self.video: VideoCapture = VideoCapture(0)
        self.update_frame()

    def update_frame(self) -> None:
        """
        Updates current frame of the screen.
        Allows the application of AI through each frame by single_frame.
        """
        # Capture single_frame-by-single_frame
        has_returned, single_frame = self.video.read()

        if has_returned:
            humans = self.pose_estimator.inference(single_frame, resize_to_default=(self.model_width > 0 and self.model_height > 0), upsample_size=self.resize_out_ratio)
            movement_state: str = self.detect_movement(humans)

            single_frame = TfPoseEstimator.draw_humans(single_frame, humans, imgcopy=False)
            self.video_frame.configure(text=movement_state)

            # Convert the single_frame to RGB
            single_frame = cvtColor(single_frame, COLOR_BGR2RGB)

            # Convert to a PIL image
            pil_image = Image.fromarray(single_frame)

            # Calculate aspect ratios
            label_aspect = self.video_frame_size[0] / self.video_frame_size[1]
            image_aspect = pil_image.width / pil_image.height

            if label_aspect > image_aspect:
                # Wider label, fit by height and crop width
                new_height = self.video_frame_size[1]
                new_width = int(new_height * image_aspect)
            else:
                # Taller label, fit by width and crop height
                new_width = self.video_frame_size[0]
                new_height = int(new_width / image_aspect)

            # Resize the image while maintaining aspect ratio
            pil_image = pil_image.resize((new_width, new_height))

            # Crop the image to fit the label size
            left = (pil_image.width - self.video_frame_size[0]) // 2
            top = (pil_image.height - self.video_frame_size[1]) // 2
            right = left + self.video_frame_size[0]
            bottom = top + self.video_frame_size[1]
            pil_image = pil_image.crop((left, top, right, bottom))

            # Convert to CTkImage
            recomputed_frame: CTkImage = CTkImage(light_image=pil_image, size=self.video_frame_size)

            # Update the label with the new image
            self.video_frame.imgtk = recomputed_frame
            self.video_frame.configure(image=recomputed_frame)

        # Schedule the next single_frame update
        self.video_frame.after(10, self.update_frame)

    def update_button_colors(self) -> None:
        """
        Updates the colors of the buttons based on the selected movement
        """
        for button, movement in zip(self.buttons, self.movements):
            new_color: str = "green" if movement == self.preferred_movement else get_color("CTkButton") # type: ignore
            button.configure(fg_color = new_color)

        self.update_idletasks()

    def set_new_movement(self, movement: Movement) -> None:
        """
        Sets the value for the new preferred movement
        """
        self.preferred_movement = movement
        self.update_button_colors()

    def detect_movement(self, humans: List[Human]) -> str:
        """
        Detect the movement state (sitting, standing, jumping, walking, running) based on human pose keypoints.

        Args:
            humans (List[Human]): List of detected humans with body parts.

        Returns:
            str: The detected movement state: "Sitting", "Standing", "Jumping", or "Unknown".
        """
        for human in humans:
            keypoints = {
                index: (part.x, part.y)
                for index, part in human.body_parts.items()
            }

            # Debugging: print the keypoints
            print(f"Detected Keypoints: {keypoints}")

            # Check if required keypoints are available
            required_points = [8, 9, 11, 12]  # Adjust as needed for your model
            if not all(point in keypoints for point in required_points):
                return "Cannot detect the whole body"

            # Get key positions
            hip_right = keypoints[8]
            knee_right = keypoints[9]
            hip_left = keypoints[11]
            knee_left = keypoints[12]

            # Calculate conditions
            standing_condition = hip_right[1] < knee_right[1] and hip_left[1] < knee_left[1]
            sitting_condition = hip_right[1] > knee_right[1] and hip_left[1] > knee_left[1]
            jumping_condition = hip_right[1] < 0.8 and hip_left[1] < 0.8
            walking_condition = (
                abs(keypoints.get(10, (0, 0))[0] - keypoints.get(13, (0, 0))[0]) < 0.2
                and knee_right[1] > hip_right[1]
                and knee_left[1] > hip_left[1]
            )
            running_condition = (
                abs(keypoints.get(10, (0, 0))[0] - keypoints.get(13, (0, 0))[0]) >= 0.2
                and knee_right[1] > hip_right[1]
                and knee_left[1] > hip_left[1]
            )

            # Match detected conditions
            movements = {
                Movement.STANDING: standing_condition,
                Movement.SITTING: sitting_condition,
                Movement.JUMPING: jumping_condition,
                Movement.WALKING: walking_condition,
                Movement.RUNNING: running_condition,
            }

            detected_movements = [movement for movement, condition in movements.items() if condition]

            if self.preferred_movement != Movement.ALL:
                if self.preferred_movement in detected_movements:
                    return self.preferred_movement.value
                return "No Actions Found"

            if detected_movements:
                return detected_movements[0].value

        return "No Actions Found"

    def on_closing(self):
        """
        Release the video capture when closing the app
        """
        if self.video.isOpened():
            self.video.release()

        self.destroy()

# Create and run the app
app: App = App()
app.protocol("WM_DELETE_WINDOW", app.on_closing)
app.mainloop()
