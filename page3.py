import tkinter as tk
import tkinter as tk
import os
import tkinter as tk
import webbrowser
class Page3(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.background_image = tk.PhotoImage(file="bg (9).png")  # Replace with your image path
        # Create a label to display the background image
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        def open_url(url):
            webbrowser.open_new(url)

        # Create the main window


        # Define the URLs and button labels
        links = {
            "Symtoms and treatment": "https://www.upmc.com/services/orthopaedics/conditions/foot-ulcer",
            "Conditions": "https://www.upmc.com/services/orthopaedics/conditions",
            "Diabitic Foot ulcer":"https://medlineplus.gov/ency/patientinstructions/000077.htm",
        }

        # Create buttons for each link
        for label, url in links.items():
            button = tk.Button(self, text=label, fg="blue", cursor="hand2", font=("Arial", 12, "underline"),
                               command=lambda url=url: open_url(url))
            button.pack(pady=5)  # Add some vertical padding for each button
