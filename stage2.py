import tkinter as tk

# Create the main application window
root = tk.Tk()
root.title("Second Stage (0.4 to 0.6 Range)")

# Create a Text widget to display or edit text
text_widget = tk.Text(root, width=40, height=10, font=("Arial", 12))
text_widget.pack(pady=20)

# Insert some text into the Text widget
text_widget.insert("1.0", "Second Stage (0.4 to 0.6 Range): \n"
                          "Images show moderate color intensification"
                          " (darker hues), irregular edges with noticeable angular variations, and a "
                          "larger area of the affected region. Gradients are steeper, suggesting progression.")

# Make the Text widget read-only (optional)
text_widget.config(state="disabled")

# Run the application
root.mainloop()
