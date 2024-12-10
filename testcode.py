import tkinter as tk
from tkinter import PhotoImage


class MyApp(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Load the background image
        self.background_image = PhotoImage(file="bg.png")  # Replace with your image path
        # Create a label to display the background image
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Configure the rest of the frame
        self.configure(bg="white")

        # Place other widgets on top of the background
        self.create_widgets()

    def create_widgets(self):
        # Example of a button with a link
        button = tk.Button(self, text="Open Google", command=lambda: self.open_url("https://www.google.com"))
        button.pack(pady=10)

    def open_url(self, url):
        import webbrowser
        webbrowser.open_new(url)


# Set up the main application window
root = tk.Tk()
root.title("App with Background Image")

# Set the window size to match the background image (optional)
root.geometry("800x600")  # Replace with your desired dimensions

# Instantiate the frame and add it to the window
app = MyApp(root)
app.pack(fill="both", expand=True)

# Start the main loop
root.mainloop()
