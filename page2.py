import tkinter as tk
import tkinter as tk
import os
from tkinter import messagebox
from tkinter import filedialog, messagebox
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import os, shutil
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
class Page2(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.background_image = tk.PhotoImage(file="bg (9).png")  # Replace with your image path
        # Create a label to display the background image
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        uploaded_file_path = ""

        # Function to classify image and display confidence score and class name
        def classify():
            global uploaded_file_path
            # --------------------------
            model = tf.keras.models.load_model('keras_Model.h5')

            # Load the label names
            with open('labels.txt', 'r') as f:
                label_names = f.read().splitlines()

            # Load and preprocess the image
            img_path = uploaded_file_path

            img = image.load_img(img_path, target_size=(150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            # Predict the class of the image
            predictions = model.predict(x)
            class_index = np.argmax(predictions[0])
            class_name_predicted = label_names[class_index]
            print(class_name_predicted)
            pclass = str(class_name_predicted)

            # -----------------------------
            if (pclass == "foot"):

                if uploaded_file_path == "":
                    confidence_label.configure(text='No image uploaded!', fg='red')
                    class_label.configure(text='No prediction available', fg='red')
                    return

                path = "output"
                if os.path.exists(path):
                    # If directory exists, delete it
                    print(f"Directory '{path}' exists. Deleting it.")
                    shutil.rmtree(path)
                    print(f"Directory '{path}' deleted.")
                else:
                    # If directory doesn't exist, create it
                    print(f"Directory '{path}' does not exist. Creating it.")
                    os.makedirs(path)
                    print(f"Directory '{path}' created.")
                model = YOLO("best.pt")
                results = model.predict(source=uploaded_file_path, project="output", save=True, save_txt=True, conf=0.5,line_thickness=6)

                # Extract the top prediction and confidence score
                result = results[0]
                boxes = result.boxes  # YOLO v8 returns boxes and their confidences
                if len(boxes) > 0:
                    # Get the highest confidence prediction
                    top_index = np.argmax(boxes.conf.numpy())  # Get the index of the highest confidence
                    top_confidence = boxes.conf[top_index].item()  # Get the confidence score
                    predicted_class = model.names[int(boxes.cls[top_index].item())]  # Get the class name

                    # Display result image
                    filename = os.path.basename(uploaded_file_path).split('/')[-1]

                    # im = Image.open(r"output/predict/" + filename)
                    # thumbnail_size = (500, 500)
                    # im.show()

                    uploaded = Image.open("output/predict/" + filename)

                    uploaded.thumbnail(
                        ((self.winfo_width() / 2), (self.winfo_height() / 2)))  # Adjust thumbnail size for better view
                    im = ImageTk.PhotoImage(uploaded)
                    resultimg.configure(image=im)
                    resultimg.image = im

                    # Display confidence score and predicted class
                    confidence_label.configure(text=f'Confidence: {top_confidence:.2f}', fg='green')
                    class_label.configure(text=f'Predicted Class: {predicted_class}', fg='green')


                    if predicted_class=="Stage 1_Ulcer":
                        os.system("python stage1.py")
                    if predicted_class=="Stage 2_Ulcer":
                        os.system("python stage2.py")
                    if predicted_class == "Stage 3_Ulcer":
                        os.system("python stage3.py")
                    if predicted_class=="Final_Stage_Ulcer":
                        os.system("python stage4.py")
                else:
                    confidence_label.configure(text='No detections', fg='red')
                    class_label.configure(text='No prediction available', fg='red')
            else:
                messagebox.showinfo("info", "Wrong Input")

        # Upload image function
        def upload_image():
            global uploaded_file_path
            try:
                uploaded_file_path = filedialog.askopenfilename(title="Select an image file",
                                                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"),
                                                                           ("JPEG files", "*.jpg;*.jpeg"), ("PNG files",
                                                                                                            "*.png")])  # Store the file path
                if uploaded_file_path:  # If a file was selected
                    uploaded = Image.open(uploaded_file_path)
                    uploaded.thumbnail(
                        ((self.winfo_width() / 3), (self.winfo_height() / 3)))  # Resize image for better layout
                    im = ImageTk.PhotoImage(uploaded)
                    sign_image.configure(image=im)
                    sign_image.image = im
                    label.configure(text='Image uploaded! Now click "Classify Image" to classify it.', fg='black')
                    confidence_label.configure(text='Confidence: N/A')  # Reset confidence label
                    class_label.configure(text='Predicted Class: N/A')  # Reset class label

                    uploaded = Image.open("unknown/Untitled.png")

                    uploaded.thumbnail(
                        ((self.winfo_width() / 2), (self.winfo_height() / 2)))  # Adjust thumbnail size for better view
                    uim = ImageTk.PhotoImage(uploaded)
                    resultimg.configure(image=uim)
                    resultimg.image = uim
                else:
                    label.configure(text='No image selected.', fg='red')
            except Exception as e:
                print(f"Error: {e}")

        # Instruction label
        label = Label(self, font=('Helvetica', 12), bg='#F0F0F0',
                      fg='#555555')  # Light gray background with soft dark text
        label.grid(row=6, column=0, columnspan=2, pady=20)

        # Label for uploaded image
        sign_image = Label(self, bg='#F0F0F0')
        sign_image.grid(row=3, column=0, padx=30, pady=20)  # Left image in the third row, first column, with padding

        # Label for result image
        resultimg = Label(self, bg='#F0F0F0')
        resultimg.grid(row=3, column=1, padx=30, pady=20)  # Right image in the third row, second column, with padding

        # Confidence score label
        confidence_label = Label(self, text="Confidence: N/A", font=('Helvetica', 14), bg='#F0F0F0', fg='#000000')
        confidence_label.grid(row=4, column=0, columnspan=2, pady=10)  # Center the confidence score

        # Predicted class label
        class_label = Label(self, text="Predicted Class: N/A", font=('Helvetica', 14), bg='#F0F0F0', fg='#000000')
        class_label.grid(row=5, column=0, columnspan=2, pady=10)  # Center the predicted class

        # Upload Image Button
        upload = Button(self, text="Upload Image", command=upload_image, padx=15, pady=10)
        upload.configure(background='#5E81AC', foreground='white', font=('Helvetica', 14, 'bold'), relief="flat",
                         borderwidth=0)  # Same color for consistency
        upload.grid(row=7, column=0, padx=20, pady=10)  # First button, side by side in row 7

        # Classify Image Button
        classify_button = Button(self, text="Classify Image", command=classify, padx=15, pady=10)
        classify_button.configure(background='#5E81AC', foreground='white', font=('Helvetica', 14, 'bold'),
                                  relief="flat", borderwidth=0)  # Same color for consistency
        classify_button.grid(row=7, column=1, padx=20, pady=10)  # Second button, side by side in row 7

        # Heading with clean, large font
        heading = Label(self, text="Foot Ulcer Detection", pady=20, font=('Helvetica', 30, 'bold'))
        heading.configure(background='#F0F0F0', foreground='#2E3440')  # Dark gray heading for contrast
        heading.grid(row=0, column=0, columnspan=2)

        # Center the layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_rowconfigure(6, weight=1)
        self.grid_rowconfigure(7, weight=1)

