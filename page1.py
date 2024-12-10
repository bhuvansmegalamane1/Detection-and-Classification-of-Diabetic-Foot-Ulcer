import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import PIL for handling images
from PIL import ImageTk, Image

class Page1(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.background_image = tk.PhotoImage(file="bg (9).png")  # Replace with your image path
        # Create a label to display the background image
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Title label
        title = tk.Label(self, text="Foot Ulcer", font=("Arial", 24), bg="white")
        title.pack(pady=10)

        # Description label
        description = tk.Label(self, text="Foot ulceration is among the most common health issues, and its prevalence has increased recently. It is one of the major causes of amputations, particularly in patients with uncontrolled diabetes."
                                          " The lifetime risk of developing a foot ulcer in patients with diabetes is more than 33%. Diabetic foot ulcer causes a lot of morbidities and accounts for approximately two-thirds of all United States"
                                          " non-traumatic amputations. Infections in these patients are thought to be limb-threatening conditions. This activity outlines and reviews the evaluation, treatment, and management of foot ulceration"
                                          " and reviews the role of interprofessional teams in evaluating and treating patients with this condition",
                               font=("Arial", 14), bg="white", wraplength=500, justify="left")
        description.pack(pady=10)

        # Displaying an image
        try:
            # Open the image file and resize it
            image = Image.open("11.png")  # Replace with your image path
            image = image.resize((250, 150), Image.LANCZOS)
            image_tk = ImageTk.PhotoImage(image)

            # Create a label to display the image
            image_label = tk.Label(self, image=image_tk, bg="white")
            image_label.image = image_tk  # Keep a reference to avoid garbage collection
            image_label.pack(pady=10)
        except Exception as e:
            error_label = tk.Label(self, text="Image could not be loaded.", font=("Arial", 12), bg="white", fg="red")
            error_label.pack(pady=10)
            print(f"Error loading image: {e}")

        # Additional content or links
        more_info = tk.Label(self, text="Explore more by navigating through the sidebar!",
                             font=("Arial", 12), bg="white")
        more_info.pack(pady=20)
