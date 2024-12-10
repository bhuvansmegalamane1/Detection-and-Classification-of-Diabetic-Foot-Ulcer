import tkinter as tk

# Create the main application window
root = tk.Tk()
root.title("Final Stage (0.8 to 1 Range)")

# Create a Text widget to display or edit text
text_widget = tk.Text(root, width=40, height=10, font=("Arial", 12))
text_widget.pack(pady=20)

# Insert some text into the Text widget



text_widget.insert("1.0", "Final Stage (0.8 to 1 Range): \n "
                          "The final stage of a foot ulcer is typically considered "
                          "Stage 4, where the ulcer has progressed to involve deeper "
                          "tissues like muscles, tendons, and bone, "
                          "often resulting in gangrene (dead tissue) and "
                          "requiring immediate medical intervention, "
                          "potentially including amputation if left untreated")

# Make the Text widget read-only (optional)
text_widget.config(state="disabled")

# Run the application
root.mainloop()
