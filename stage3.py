import tkinter as tk

# Create the main application window
root = tk.Tk()
root.title("Third Stage (0.7 to 0.8 Range)")

# Create a Text widget to display or edit text
text_widget = tk.Text(root, width=40, height=10, font=("Arial", 12))
text_widget.pack(pady=20)

# Insert some text into the Text widget
text_widget.insert("1.0", "Third Stage (0.7 to 0.8 Range): \n "
                          "Severe ulceration is characterized"
                          " by dark or black color tones, highly irregular edges with sharp "
                          "angular changes, and extensive pixel coverage. Texture becomes coarse, "
                          "and edge sharpness is more pronounced, indicating advanced progression.")

# Make the Text widget read-only (optional)
text_widget.config(state="disabled")

# Run the application
root.mainloop()
