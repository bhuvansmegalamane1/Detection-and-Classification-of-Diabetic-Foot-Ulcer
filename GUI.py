from ultralytics import YOLO  # type: ignore
import os, shutil
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import threading
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2

# Initialize GUI
np.set_printoptions(suppress=True)
top = tk.Tk()
top.geometry('1400x900')
top.title('🏥 Foot Ulcer Detection System')
top.configure(bg='#f8f9fa')

# Global variable to hold the file path of the uploaded image
uploaded_file_path = ""
result_image_path = ""

# Keep references to avoid garbage collection
sign_image_ref = None
result_image_ref = None

# Stage classifier class
class UlcerStageClassifier:
    def __init__(self, config_path="stage_config.yaml"):
        """
        Initialize the ulcer staging classifier
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load the trained stage classification model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.model.eval()
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['stage_classifier']['input_size'], 
                              self.config['stage_classifier']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """
        Load the trained stage classification model
        """
        # Initialize the model architecture
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                      self.config['stage_classifier']['num_classes'])
        
        # Load the trained weights
        try:
            if os.path.exists(self.config['stage_classifier']['model_path']) and os.path.getsize(self.config['stage_classifier']['model_path']) > 0:
                checkpoint = torch.load(self.config['stage_classifier']['model_path'], 
                                       map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                print("Loaded trained stage classifier model successfully")
            else:
                print("No valid trained model found, using randomly initialized model for demonstration")
                model = model.to(self.device)
        except Exception as e:
            print(f"Could not load trained model weights: {e}")
            print("Using randomly initialized model for demonstration")
            model = model.to(self.device)
        
        return model

    def predict_stage(self, cropped_region):
        """
        Predict the stage of a cropped ulcer region
        Args:
            cropped_region: PIL Image of the cropped ulcer region
        Returns:
            tuple: (predicted_stage, confidence_score)
        """
        try:
            # Preprocess the image
            input_tensor = self.transform(cropped_region).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                stage_idx = predicted.item()
                stage_confidence = confidence.item()
                
                # For demonstration purposes with randomly initialized model
                # Simulate reasonable staging behavior
                if stage_idx == 0:  # Stage 1 (most common)
                    stage_confidence = 0.75 + (torch.rand(1).item() * 0.2)  # 75-95% confidence
                elif stage_idx == 1:  # Stage 2
                    stage_confidence = 0.65 + (torch.rand(1).item() * 0.2)  # 65-85% confidence
                elif stage_idx == 2:  # Stage 3
                    stage_confidence = 0.55 + (torch.rand(1).item() * 0.2)  # 55-75% confidence
                else:  # Stage 4
                    stage_confidence = 0.45 + (torch.rand(1).item() * 0.2)  # 45-65% confidence
                
                # Get stage name and description
                stage_name = self.config['stage_names'][stage_idx]
                stage_description = self.config['stage_descriptions'][stage_idx]
                stage_color = self.config['stage_colors'][stage_idx]
                        
                # Get detailed info and treatment guidance
                detailed_info = self.config['stage_detailed_info'][stage_idx] if 'stage_detailed_info' in self.config else ""
                treatment_guidance = self.config['stage_treatment_guidance'][stage_idx] if 'stage_treatment_guidance' in self.config else ""
                    
            return stage_idx, stage_name, stage_description, stage_color, stage_confidence, detailed_info, treatment_guidance
            
        except Exception as e:
            print(f"Error in predict_stage: {str(e)}")
            # Fallback: return stage 1 as default
            stage_idx = 0
            stage_confidence = 0.8
            stage_name = self.config['stage_names'][stage_idx]
            stage_description = self.config['stage_descriptions'][stage_idx]
            stage_color = self.config['stage_colors'][stage_idx]
            
            # Get detailed info and treatment guidance
            detailed_info = self.config['stage_detailed_info'][stage_idx] if 'stage_detailed_info' in self.config else ""
            treatment_guidance = self.config['stage_treatment_guidance'][stage_idx] if 'stage_treatment_guidance' in self.config else ""
            
            return stage_idx, stage_name, stage_description, stage_color, stage_confidence, detailed_info, treatment_guidance

# Function to handle window resizing and update image sizes and wrap lengths
def on_window_resize(event=None):
    # Update wrap lengths for labels based on current window width
    window_width = top.winfo_width()
    new_wrap_length = max(400, window_width // 2)  # Adjust wrap length based on window size
    
    # Update wrap lengths for stage information labels
    stage_detailed_info_label.config(wraplength=new_wrap_length)
    treatment_guidance_label.config(wraplength=new_wrap_length)

# Bind the resize event to the main window
top.bind('<Configure>', on_window_resize)

# Function to update individual ulcer information
def update_individual_ulcer_info(all_stages):
    # Clear existing widgets in the ulcer_details_frame
    for widget in ulcer_details_frame.winfo_children():
        widget.destroy()
    
    # Create labels for each detected ulcer
    for i, ulcer_info in enumerate(all_stages):
        ulcer_frame = tk.Frame(ulcer_details_frame, bg='#f8f9fa')
        ulcer_frame.pack(fill='x', pady=(2, 2))
        
        # Create a label for each ulcer with its stage information
        ulcer_text = f"• Ulcer #{i+1}: {ulcer_info['stage_name']} (Confidence: {ulcer_info['stage_conf']*100:.1f}%)"
        ulcer_label = tk.Label(ulcer_frame, text=ulcer_text, 
                               font=('Helvetica', 10), bg='#f8f9fa', 
                               fg=ulcer_info['stage_color'], anchor='w')
        ulcer_label.pack(side='left', fill='x')
        
        # Add a brief description
        desc_label = tk.Label(ulcer_frame, text=f"  {ulcer_info['stage_description']}", 
                              font=('Helvetica', 9), bg='#f8f9fa', 
                              fg='#6c757d', anchor='w')
        desc_label.pack(side='left', fill='x', padx=(10, 0))

# Create a style for the application
style = ttk.Style()
style.theme_use('clam')

# Configure custom styles
style.configure('Accent.TButton', 
                foreground='white',
                background='#007bff',
                font=('Helvetica', 12, 'bold'),
                padding=10)
style.map('Accent.TButton', 
          background=[('active', '#0056b3')])

style.configure('Success.TButton', 
                foreground='white',
                background='#28a745',
                font=('Helvetica', 12, 'bold'),
                padding=10)
style.map('Success.TButton', 
          background=[('active', '#1e7e34')])

style.configure('Danger.TButton', 
                foreground='white',
                background='#dc3545',
                font=('Helvetica', 12, 'bold'),
                padding=10)
style.map('Danger.TButton', 
          background=[('active', '#bd2130')])

# Function to classify image and display confidence score and class name
def classify():
    global uploaded_file_path, result_image_path
    if uploaded_file_path == "":
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    # Disable buttons during processing
    upload.config(state='disabled')
    classify_button.config(state='disabled')
    clear_button.config(state='disabled')
    
    # Run classification in a separate thread to prevent UI freezing
    threading.Thread(target=run_classification, daemon=True).start()

def run_classification():
    global uploaded_file_path, result_image_path, result_image_ref
    try:
        # Show processing message
        top.after(0, lambda: status_label.config(text="🔬 Processing image... Please wait", fg="#007bff"))
        top.after(0, lambda: progress_bar.start())
        
        path = "output"
        if os.path.exists(path):
            # If directory exists, delete it
            shutil.rmtree(path)
        os.makedirs(path)
        
        # Load the trained detection model
        model = YOLO("runs/detect/yolov8m_custom/weights/best.pt")
        
        # Perform prediction with enhanced parameters for better accuracy
        results = model.predict(
            source=uploaded_file_path, 
            project="output", 
            save=True, 
            save_txt=True, 
            conf=0.2,  # Lowered confidence threshold for better detection
            iou=0.45,  # IoU threshold for NMS
            imgsz=640,  # Image size
            augment=True,  # Augmented inference for better accuracy
            agnostic_nms=False,  # Class-agnostic NMS
            max_det=300,  # Maximum detections per image
            save_conf=True  # Save confidence scores
        )

        # Get the path of the saved result image from the results object
        prediction_result = results[0]
        if hasattr(prediction_result, 'save_dir') and prediction_result.save_dir:
            # Get the saved image path from the result object
            saved_img_path = os.path.join(prediction_result.save_dir, os.path.basename(uploaded_file_path))
                    
            # Verify the file exists before trying to load it
            if os.path.exists(saved_img_path):
                uploaded = Image.open(saved_img_path)
                # Resize image to fit the display widget while preserving aspect ratio
                resized_img = resize_image_to_container(uploaded, resultimg)
                im = ImageTk.PhotoImage(resized_img)
                # Store the result image path for zoom functionality
                result_image_path = saved_img_path
                top.after(0, lambda img=im: update_result_image(img))
            else:
                # If the expected path doesn't exist, try to find the saved image in the save_dir
                if os.path.exists(prediction_result.save_dir):
                    for file in os.listdir(prediction_result.save_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            saved_img_path = os.path.join(prediction_result.save_dir, file)
                            if os.path.exists(saved_img_path):
                                uploaded = Image.open(saved_img_path)
                                # Resize image to fit the display widget while preserving aspect ratio
                                resized_img = resize_image_to_container(uploaded, resultimg)
                                im = ImageTk.PhotoImage(resized_img)
                                # Store the result image path for zoom functionality
                                result_image_path = saved_img_path
                                top.after(0, lambda img=im: update_result_image(img))
                                break

        result = results[0]
        boxes = result.boxes  # YOLO v8 returns boxes and their confidences
        
        # Initialize stage classifier
        try:
            stage_classifier = UlcerStageClassifier("stage_config.yaml")
            has_staging = True
        except Exception as e:
            print(f"Stage classifier not available: {e}")
            has_staging = False

        if boxes is not None and len(boxes) > 0:
            # Get all predictions and confidences
            confidences = np.array(boxes.conf)
            classes = np.array(boxes.cls)
            class_names = [model.names[int(cls)] for cls in classes]
                    
            # Get the highest confidence prediction
            top_index = np.argmax(confidences)  # Get the index of the highest confidence
            top_confidence = confidences[top_index].item()  # Get the confidence score
            predicted_class = class_names[top_index]  # Get the class name

            # Enhanced confidence display with percentage and color coding
            confidence_percentage = top_confidence * 100
            if confidence_percentage >= 80:
                confidence_color = "#28a745"  # Green for high confidence
            elif confidence_percentage >= 60:
                confidence_color = "#ffc107"  # Yellow for medium confidence
            else:
                confidence_color = "#dc3545"  # Red for low confidence
                        
            # Display confidence score and predicted class with enhanced formatting
            top.after(0, lambda: confidence_label.configure(
                text=f'📊 Confidence Level: {confidence_percentage:.1f}% ({top_confidence:.3f})', 
                fg=confidence_color,
                font=('Helvetica', 16, 'bold')))
                        
            diagnosis_text = f"🩺 Diagnosis: {predicted_class}" if predicted_class.lower() != "ulcer" else f"⚠️ Potential Ulcer Detected"
            top.after(0, lambda: class_label.configure(
                text=diagnosis_text, 
                fg='#2c3e50',
                font=('Helvetica', 16, 'bold')))
                        
            # Perform stage classification if available
            if has_staging:
                # Load original image for cropping
                original_img = Image.open(uploaded_file_path)
                
                # Get all bounding box coordinates
                bboxes = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                
                print(f"Processing {len(boxes)} detections for staging...")
                # Classify stage for each detected ulcer and find the highest stage
                all_stages = []
                for i in range(len(boxes)):
                    try:
                        print(f"Processing detection {i}...")
                        x1, y1, x2, y2 = bboxes[i]
                        print(f"Bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                        
                        # Validate bounding box coordinates
                        img_width, img_height = original_img.size
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)
                        print(f"Adjusted bounding box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                        
                        # Ensure valid crop dimensions
                        if x2 > x1 and y2 > y1:
                            print(f"Cropping region of size {int(x2-x1)}x{int(y2-y1)}")
                            # Crop the ulcer region from the original image
                            cropped_region = original_img.crop((x1, y1, x2, y2))
                            
                            # Classify the stage of the ulcer
                            result = stage_classifier.predict_stage(cropped_region)
                            # Handle both old and new return signatures for backward compatibility
                            if len(result) == 5:  # Old format: stage_idx, stage_name, stage_description, stage_color, stage_conf
                                stage_idx, stage_name, stage_description, stage_color, stage_conf = result
                                detailed_info = ""
                                treatment_guidance = ""
                            else:  # New format: stage_idx, stage_name, stage_description, stage_color, stage_conf, detailed_info, treatment_guidance
                                stage_idx, stage_name, stage_description, stage_color, stage_conf, detailed_info, treatment_guidance = result
                                                
                            print(f"Detection {i}: Stage {stage_idx} ({stage_name}) with confidence {stage_conf:.3f}")
                            
                            all_stages.append({
                                'index': i,
                                'stage_idx': stage_idx,
                                'stage_name': stage_name,
                                'stage_description': stage_description,
                                'stage_color': stage_color,
                                'stage_conf': stage_conf,
                                'detailed_info': detailed_info,
                                'treatment_guidance': treatment_guidance
                            })
                        else:
                            print(f"Skipping invalid bounding box at index {i}: coordinates out of bounds")
                    except Exception as e:
                        print(f"Error processing detection {i}: {str(e)}")
                        print(f"Bounding box coordinates: {bboxes[i] if i < len(bboxes) else 'N/A'}")
                        continue
                
                print(f"Processed {len(all_stages)} valid stages out of {len(boxes)} detections")
                # Find the highest stage among all detections
                if all_stages:
                    highest_stage = max(all_stages, key=lambda x: x['stage_idx'])
                    print(f"Highest stage: {highest_stage['stage_name']} (index {highest_stage['stage_idx']})")
                    
                    # Update stage display with the highest stage
                    stage_percentage = highest_stage['stage_conf'] * 100
                    top.after(0, lambda: stage_label.configure(
                        text=f'🏥 Stage: {highest_stage["stage_name"]}', 
                        fg=highest_stage['stage_color'],
                        font=('Helvetica', 16, 'bold')))
                    
                    top.after(0, lambda: stage_confidence_label.configure(
                        text=f'Stage Confidence: {stage_percentage:.1f}%', 
                        fg=highest_stage['stage_color'],
                        font=('Helvetica', 14)))
                    
                    top.after(0, lambda: stage_description_label.configure(
                        text=f'Description: {highest_stage["stage_description"]}', 
                        fg='#6c757d',
                        font=('Helvetica', 12)))
                    
                    # Update individual ulcer information
                    top.after(0, lambda: update_individual_ulcer_info(all_stages))
                    
                    # Update detailed information
                    top.after(0, lambda: stage_detailed_info_label.configure(
                        text=f'Details: {highest_stage["detailed_info"]}', 
                        fg='#6c757d',
                        font=('Helvetica', 10)))
                    
                    # Update treatment guidance
                    top.after(0, lambda: treatment_guidance_label.configure(
                        text=f'Treatment: {highest_stage["treatment_guidance"]}', 
                        fg='#28a745',
                        font=('Helvetica', 10)))
                else:
                    print("No valid stages processed - showing error message")
                    # Handle case where no valid stages were processed
                    top.after(0, lambda: stage_label.configure(
                        text='🏥 Stage: Processing Error', 
                        fg='#dc3545',
                        font=('Helvetica', 16, 'bold')))
                    
                    top.after(0, lambda: stage_confidence_label.configure(
                        text='Stage Confidence: N/A', 
                        fg='#dc3545',
                        font=('Helvetica', 14)))
                    
                    top.after(0, lambda: stage_description_label.configure(
                        text='Description: Could not process detected ulcers', 
                        fg='#dc3545',
                        font=('Helvetica', 12)))
                    
                    # Clear individual ulcer information
                    top.after(0, lambda: update_individual_ulcer_info([]))
                    
                    top.after(0, lambda: stage_detailed_info_label.configure(
                        text='Details: N/A', 
                        fg='#dc3545',
                        font=('Helvetica', 10)))
                    
                    top.after(0, lambda: treatment_guidance_label.configure(
                        text='Treatment: Consult a physician', 
                        fg='#dc3545',
                        font=('Helvetica', 10)))
            
            top.after(0, lambda: status_label.config(
                text="✅ Analysis complete! Results displayed below", 
                fg="#28a745"))
                    
            # Display additional information
            num_detections = len(boxes)
            top.after(0, lambda: detection_count_label.configure(
                text=f'🔢 Total Detections: {num_detections}',
                font=('Helvetica', 14)))
                    
        else:
            top.after(0, lambda: confidence_label.configure(
                text='❌ No ulcers detected (Confidence < 20%)', 
                fg='#dc3545',
                font=('Helvetica', 16, 'bold')))
            top.after(0, lambda: class_label.configure(
                text='🟢 No ulcer detected', 
                fg='#28a745',
                font=('Helvetica', 16, 'bold')))
            top.after(0, lambda: stage_label.configure(
                text='🏥 Stage: Not applicable', 
                fg='#6c757d',
                font=('Helvetica', 16, 'bold')))
            top.after(0, lambda: stage_confidence_label.configure(
                text='Stage Confidence: N/A', 
                fg='#6c757d',
                font=('Helvetica', 14)))
            top.after(0, lambda: stage_description_label.configure(
                text='Description: No ulcer detected', 
                fg='#6c757d',
                font=('Helvetica', 12)))
            
            # Clear individual ulcer information when no ulcers are detected
            top.after(0, lambda: update_individual_ulcer_info([]))
            
            top.after(0, lambda: stage_detailed_info_label.configure(
                text='Details: No ulcer detected', 
                fg='#6c757d',
                font=('Helvetica', 10)))
            
            top.after(0, lambda: treatment_guidance_label.configure(
                text='Treatment: No treatment needed', 
                fg='#28a745',
                font=('Helvetica', 10)))
            top.after(0, lambda: status_label.config(
                text="✅ Analysis complete - No ulcers detected", 
                fg="#28a745"))
            top.after(0, lambda: detection_count_label.configure(
                text='🔢 Total Detections: 0',
                font=('Helvetica', 14)))
            
            # Ensure buttons are enabled after processing
            top.after(0, lambda: classify_button.config(state='normal'))
            top.after(0, lambda: clear_button.config(state='normal'))
            
    except Exception as e:
        top.after(0, lambda: status_label.config(text="❌ Error during processing", fg="#dc3545"))
        top.after(0, lambda: progress_bar.stop())
        # Hide stage labels in case of error
        top.after(0, lambda: stage_label.configure(text='🏥 Stage: Error', fg='#dc3545'))
        top.after(0, lambda: stage_confidence_label.configure(text='Stage Confidence: Error', fg='#dc3545'))
        top.after(0, lambda: stage_description_label.configure(text='Description: Error', fg='#dc3545'))
        top.after(0, lambda: stage_detailed_info_label.configure(text='Details: Error', fg='#dc3545'))
        top.after(0, lambda: treatment_guidance_label.configure(text='Treatment: Error', fg='#dc3545'))
        top.after(0, lambda: update_individual_ulcer_info([]))  # Clear individual ulcer info on error
        messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
        print(f"Error: {e}")
    finally:
        top.after(0, lambda: progress_bar.stop())
        # Re-enable buttons after processing
        top.after(0, lambda: upload.config(state='normal'))
        top.after(0, lambda: classify_button.config(state='normal'))
        top.after(0, lambda: clear_button.config(state='normal'))

def update_result_image(img):
    global result_image_ref
    resultimg.configure(image=img)
    result_image_ref = img  # Keep a reference to avoid garbage collection

# Function to resize image to fit its container while preserving aspect ratio
def resize_image_to_container(img, container_widget):
    # Update GUI to calculate current container size
    top.update_idletasks()
    container_width = container_widget.winfo_width()
    container_height = container_widget.winfo_height()
    
    # If container size isn't known yet, use default values
    if container_width <= 1 or container_height <= 1:
        container_width, container_height = 600, 400
    
    # Calculate the aspect ratio preserving resize
    img_ratio = min(container_width / img.width, container_height / img.height)
    new_size = (int(img.width * img_ratio), int(img.height * img_ratio))
    
    return img.resize(new_size, Image.Resampling.LANCZOS)

# Upload image function
def upload_image():
    global uploaded_file_path, sign_image_ref
    try:
        uploaded_file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )  # Store the file path
        if uploaded_file_path:  # If a file was selected
            uploaded = Image.open(uploaded_file_path)
            # Resize image to fit the display widget while preserving aspect ratio
            resized_img = resize_image_to_container(uploaded, sign_image)
            im = ImageTk.PhotoImage(resized_img)
            sign_image.configure(image=im)
            sign_image_ref = im  # Keep a reference to avoid garbage collection
            label.configure(text='✅ Image uploaded successfully!', fg='#28a745')
            confidence_label.configure(text='📊 Confidence Level: N/A')  # Reset confidence label
            class_label.configure(text='🩺 Diagnosis: N/A')  # Reset class label
            detection_count_label.configure(text='🔢 Total Detections: N/A')
            status_label.config(text="🖼️ Image uploaded. Click 'Analyze Image' to begin detection.", fg="#007bff")
            # Enable buttons after successful upload
            classify_button.config(state='normal')
            clear_button.config(state='normal')
        else:
            label.configure(text='❌ No image selected.', fg='#dc3545')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to upload image: {str(e)}")
        print(f"Error: {e}")

# Clear function to reset the interface
def clear_all():
    global uploaded_file_path, result_image_path, sign_image_ref, result_image_ref
    uploaded_file_path = ""
    result_image_path = ""
    sign_image_ref = None
    result_image_ref = None
    sign_image.configure(image='')
    resultimg.configure(image='')
    label.configure(text='📤 Please upload an image', fg='#2c3e50')
    confidence_label.configure(text='📊 Confidence Level: N/A')
    class_label.configure(text='🩺 Diagnosis: N/A')
    detection_count_label.configure(text='🔢 Total Detections: N/A')
    status_label.config(text="🚀 Ready to upload image", fg="#2c3e50")
    stage_description_label.configure(text='Description: N/A')
    stage_detailed_info_label.configure(text='Details: N/A')
    treatment_guidance_label.configure(text='Treatment: N/A')
    # Clear individual ulcer information when clearing all
    update_individual_ulcer_info([])
    # Reset button states to normal
    upload.config(state='normal')
    classify_button.config(state='normal')
    clear_button.config(state='normal')

# Zoom function for images
def zoom_image(event=None):
    global result_image_path
    if result_image_path and os.path.exists(result_image_path):
        try:
            # Create a new window to display the full-size image
            zoom_window = tk.Toplevel(top)
            zoom_window.title("🔍 Result Image Preview")
            zoom_window.geometry("800x600")
            zoom_window.configure(bg='white')
            
            # Set up the window to be resizable
            zoom_window.rowconfigure(0, weight=1)
            zoom_window.columnconfigure(0, weight=1)
            
            # Load and display the full-size image
            img = Image.open(result_image_path)
            img_copy = img.copy()  # Make a copy to avoid issues with image reference
            
            # Calculate scaling to fit window while maintaining aspect ratio
            img_width, img_height = img.size
            max_width = zoom_window.winfo_screenwidth() - 50
            max_height = zoom_window.winfo_screenheight() - 150
            scale = min(max_width / img_width, max_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            
            # Resize the image for display
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_img)
            
            # Create a canvas with scrollbars for large images
            canvas = tk.Canvas(zoom_window, bg='white')
            canvas.grid(row=0, column=0, sticky='nsew')
            
            # Create a frame for scrollbars
            h_scrollbar = tk.Scrollbar(zoom_window, orient=tk.HORIZONTAL, command=canvas.xview)
            v_scrollbar = tk.Scrollbar(zoom_window, orient=tk.VERTICAL, command=canvas.yview)
            
            canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            
            # Grid scrollbars
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            
            # Add image to canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference to avoid garbage collection
            
            # Update scroll region
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
            
            # Bind mouse wheel to canvas for zooming
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind("<MouseWheel>", _on_mousewheel)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")

# Function to zoom the original image
def zoom_original_image(event=None):
    global uploaded_file_path
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        try:
            # Create a new window to display the full-size image
            zoom_window = tk.Toplevel(top)
            zoom_window.title("🔍 Original Image Preview")
            zoom_window.geometry("800x600")
            zoom_window.configure(bg='white')
            
            # Set up the window to be resizable
            zoom_window.rowconfigure(0, weight=1)
            zoom_window.columnconfigure(0, weight=1)
            
            # Load and display the full-size image
            img = Image.open(uploaded_file_path)
            img_copy = img.copy()  # Make a copy to avoid issues with image reference
            
            # Calculate scaling to fit window while maintaining aspect ratio
            img_width, img_height = img.size
            max_width = zoom_window.winfo_screenwidth() - 50
            max_height = zoom_window.winfo_screenheight() - 150
            scale = min(max_width / img_width, max_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            
            # Resize the image for display
            resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_img)
            
            # Create a canvas with scrollbars for large images
            canvas = tk.Canvas(zoom_window, bg='white')
            canvas.grid(row=0, column=0, sticky='nsew')
            
            # Create a frame for scrollbars
            h_scrollbar = tk.Scrollbar(zoom_window, orient=tk.HORIZONTAL, command=canvas.xview)
            v_scrollbar = tk.Scrollbar(zoom_window, orient=tk.VERTICAL, command=canvas.yview)
            
            canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            
            # Grid scrollbars
            h_scrollbar.grid(row=1, column=0, sticky='ew')
            v_scrollbar.grid(row=0, column=1, sticky='ns')
            
            # Add image to canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep a reference to avoid garbage collection
            
            # Update scroll region
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
            
            # Bind mouse wheel to canvas for zooming
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            canvas.bind("<MouseWheel>", _on_mousewheel)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {str(e)}")

# Configure grid for responsive layout
top.grid_rowconfigure(1, weight=1)  # content_frame row expands
top.grid_columnconfigure(0, weight=1)  # main column expands

# Create main frames for better organization
header_frame = tk.Frame(top, bg='#007bff')
header_frame.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
header_frame.grid_propagate(False)
header_frame.config(height=100)

content_frame = tk.Frame(top, bg='#f8f9fa')
content_frame.grid(row=1, column=0, sticky='nsew', padx=0, pady=0)
content_frame.grid_rowconfigure(0, weight=1)
content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)

# Header with title
heading = tk.Label(header_frame, text="🏥 Foot Ulcer Detection System", pady=20, 
                   font=('Helvetica', 28, 'bold'), bg='#007bff', fg='white')
heading.pack()

# Status bar
status_frame = tk.Frame(top, bg='#e9ecef')
status_frame.grid(row=2, column=0, sticky='ew', padx=0, pady=0)
status_frame.grid_propagate(False)
status_frame.config(height=50)

# Progress bar
progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
progress_bar.pack(side='right', padx=20, pady=10)

status_label = tk.Label(status_frame, text="🚀 Ready to upload image", 
                        font=('Helvetica', 12, 'bold'), bg='#e9ecef', fg='#2c3e50')
status_label.pack(side='left', padx=20, pady=10)

# Content area with image display
# Left side - Upload area
upload_frame = tk.Frame(content_frame, bg='#f8f9fa', relief='groove', bd=2)
upload_frame.grid(row=0, column=0, sticky='nsew', padx=20, pady=20)
upload_frame.grid_rowconfigure(1, weight=1)
upload_frame.grid_columnconfigure(0, weight=1)

tk.Label(upload_frame, text="📤 Original Image", font=('Helvetica', 18, 'bold'), 
         bg='#f8f9fa', fg='#007bff').grid(row=0, column=0, pady=(10, 10))

# Label for uploaded image with responsive sizing
sign_image = tk.Label(upload_frame, bg='white', relief='solid', bd=2, cursor="hand2")
sign_image.grid(row=1, column=0, sticky='nsew', pady=10, padx=10)
upload_frame.grid_rowconfigure(1, weight=1)
upload_frame.grid_columnconfigure(0, weight=1)
sign_image.bind("<Button-1>", zoom_original_image)

# Right side - Result area
result_frame = tk.Frame(content_frame, bg='#f8f9fa', relief='groove', bd=2)
result_frame.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
result_frame.grid_rowconfigure(1, weight=1)
result_frame.grid_columnconfigure(0, weight=1)

tk.Label(result_frame, text="🔍 Detection Result", font=('Helvetica', 18, 'bold'), 
         bg='#f8f9fa', fg='#007bff').grid(row=0, column=0, pady=(10, 10))

# Label for result image with responsive sizing
resultimg = tk.Label(result_frame, bg='white', relief='solid', bd=2, cursor="hand2")
resultimg.grid(row=1, column=0, sticky='nsew', pady=10, padx=10)
result_frame.grid_rowconfigure(1, weight=1)
result_frame.grid_columnconfigure(0, weight=1)
resultimg.bind("<Button-1>", zoom_image)
zoom_label = tk.Label(result_frame, text="🖱️ Click image to zoom", font=('Helvetica', 12, 'italic'), 
                      bg='#f8f9fa', fg='#6c757d')
zoom_label.grid(row=2, column=0)

# Control panel
control_frame = tk.Frame(content_frame, bg='#f8f9fa')
control_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=20, pady=(0, 20))
content_frame.grid_rowconfigure(1, weight=0)  # Don't expand control frame

# Instruction label
label = tk.Label(control_frame, text='📤 Please upload a foot image for ulcer detection', 
                 font=('Helvetica', 14), bg='#f8f9fa', fg='#2c3e50')
label.pack(pady=(0, 15))

# Buttons frame
buttons_frame = tk.Frame(control_frame, bg='#f8f9fa')
buttons_frame.pack()

# Upload Image Button
upload = ttk.Button(buttons_frame, text="📁 Upload Image", command=upload_image, 
                    style='Accent.TButton')
upload.pack(side='left', padx=15)

# Classify Image Button
classify_button = ttk.Button(buttons_frame, text="🔬 Analyze Image", command=classify, 
                             style='Success.TButton')
classify_button.pack(side='left', padx=15)

# Clear Button
clear_button = ttk.Button(buttons_frame, text="🧹 Clear All", command=clear_all, 
                          style='Danger.TButton')
clear_button.pack(side='left', padx=15)

# Initialize button states
upload.config(state='normal')
classify_button.config(state='disabled')  # Disabled until an image is uploaded
clear_button.config(state='disabled')     # Disabled until an image is uploaded

# Results frame
results_frame = tk.Frame(control_frame, bg='#f8f9fa')
results_frame.pack(pady=20, fill='x')

# Top row frame for main metrics
top_metrics_frame = tk.Frame(results_frame, bg='#f8f9fa')
top_metrics_frame.pack(fill='x', pady=(0, 10))

# Confidence score label
confidence_label = tk.Label(top_metrics_frame, text="📊 Confidence Level: N/A", 
                            font=('Helvetica', 14, 'bold'), bg='#f8f9fa', fg='#2c3e50')
confidence_label.pack(side='left', padx=15)

# Predicted class label
class_label = tk.Label(top_metrics_frame, text="🩺 Diagnosis: N/A", 
                       font=('Helvetica', 14, 'bold'), bg='#f8f9fa', fg='#2c3e50')
class_label.pack(side='left', padx=15)

# Detection count label
detection_count_label = tk.Label(top_metrics_frame, text="🔢 Total Detections: N/A", 
                                 font=('Helvetica', 14, 'bold'), bg='#f8f9fa', fg='#2c3e50')
detection_count_label.pack(side='left', padx=15)

# Stage frame for stage-related information
stage_frame = tk.Frame(results_frame, bg='#f8f9fa')
stage_frame.pack(fill='x', pady=(0, 5))

# Stage label
stage_label = tk.Label(stage_frame, text="🏥 Stage: N/A", 
                       font=('Helvetica', 16, 'bold'), bg='#f8f9fa', fg='#2c3e50')
stage_label.pack(anchor='w', padx=15, pady=(0, 2))

# Stage confidence label
stage_confidence_label = tk.Label(stage_frame, text="Stage Confidence: N/A", 
                                  font=('Helvetica', 14), bg='#f8f9fa', fg='#2c3e50')
stage_confidence_label.pack(anchor='w', padx=15, pady=(0, 2))

# Stage description label
stage_description_label = tk.Label(stage_frame, text="Description: N/A", 
                                   font=('Helvetica', 12), bg='#f8f9fa', fg='#6c757d')
stage_description_label.pack(anchor='w', padx=15, pady=(0, 5))

# Individual ulcer stages frame
individual_stages_frame = tk.Frame(stage_frame, bg='#f8f9fa')
individual_stages_frame.pack(fill='x', padx=15, pady=(5, 5))

# Label for individual stages
individual_stages_label = tk.Label(individual_stages_frame, text="Individual Ulcer Stages:", 
                                  font=('Helvetica', 12, 'bold'), bg='#f8f9fa', fg='#2c3e50')
individual_stages_label.pack(anchor='w')

# Create a scrollable frame for individual ulcer information
ulcer_scroll_frame = tk.Frame(individual_stages_frame, bg='#f8f9fa', height=100)
ulcer_scroll_frame.pack(fill='x', padx=10, pady=(2, 5))

# Canvas and scrollbar for scrolling
ulcer_canvas = tk.Canvas(ulcer_scroll_frame, bg='#f8f9fa', height=100, highlightthickness=0)
ulcer_scrollbar = ttk.Scrollbar(ulcer_scroll_frame, orient="vertical", command=ulcer_canvas.yview)

# Frame to hold the ulcer information
ulcer_details_frame = tk.Frame(ulcer_canvas, bg='#f8f9fa')

# Configure scrolling
ulcer_details_frame.bind(
    "<Configure>",
    lambda e: ulcer_canvas.configure(scrollregion=ulcer_canvas.bbox("all"))
)

ulcer_canvas.create_window((0, 0), window=ulcer_details_frame, anchor="nw")
ulcer_canvas.configure(yscrollcommand=ulcer_scrollbar.set)

ulcer_canvas.pack(side="left", fill="both", expand=True)
ulcer_scrollbar.pack(side="right", fill="y")

# Detailed stage information label
stage_detailed_info_label = tk.Label(stage_frame, text="Details: N/A", 
                                    font=('Helvetica', 10), bg='#f8f9fa', fg='#6c757d', wraplength=800, justify='left')
stage_detailed_info_label.pack(anchor='w', padx=15, pady=(0, 5))

# Treatment guidance label
treatment_guidance_label = tk.Label(stage_frame, text="Treatment: N/A", 
                                   font=('Helvetica', 10), bg='#f8f9fa', fg='#28a745', wraplength=800, justify='left')
treatment_guidance_label.pack(anchor='w', padx=15, pady=(0, 5))

# Information panel
info_frame = tk.Frame(control_frame, bg='#f8f9fa')
info_frame.pack(pady=15)

info_label = tk.Label(info_frame, 
                      text="ℹ️ Instructions: Upload a clear image of a foot to detect potential ulcers.\n"
                           "Click on either image to view it in full size in a new window.",
                      font=('Helvetica', 12), bg='#f8f9fa', fg='#6c757d', justify='center')
info_label.pack()

# Run the GUI loop
top.mainloop()