import tkinter as tk

# Create the main application window
root = tk.Tk()
root.title("First Stage (0.1 to 0.3 Range)")
root.geometry("400x300")
# Create a Text widget to display or edit text
text_widget = tk.Text(root, width=40, height=10, font=("Arial", 12))
text_widget.pack(pady=20)

# Insert some text into the Text widget
text_widget.insert("1.0", "First Stage (0.1 to 0.3 Range):\n"
                          "Ulcers are detected with light-colored "
                          "regions, well-defined and smooth edges, minimal angular variation, "
                          "and a small pixel area in the image. The texture and gradient changes"
                          " are subtle, indicating early-stage classification.")

# Make the Text widget read-only (optional)
text_widget.config(state="disabled")

# Run the application
root.mainloop()
