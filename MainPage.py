import tkinter as tk
from page1 import Page1
from page2 import Page2
from page3 import Page3
from page4 import Page4
from page5 import Page5


# Initialize main window
root = tk.Tk()
root.title("Foot Ulcer")
root.geometry("1000x600")

# Define a function to switch between pages
def show_frame(frame):
    frame.tkraise()

# Create a container frame for sidebar and content
container = tk.Frame(root)
container.pack(fill="both", expand=True)

# Create the sidebar frame
sidebar = tk.Frame(container, width=150, bg="lightblue")
sidebar.pack(side="left", fill="y")

# Create the main content frame where pages will be displayed
content = tk.Frame(container, bg="white")
content.pack(side="right", fill="both", expand=True)

# Initialize pages from separate files
page1 = Page1(content)
page2 = Page2(content)
page3 = Page3(content)
page4 = Page4(content)
page5 = Page5(content)

# Stack pages on top of each other
for frame in (page1, page2, page3, page4, page5):
    frame.place(relwidth=1, relheight=1)

# Sidebar buttons for page navigation
btn1 = tk.Button(sidebar, text="Home", command=lambda: show_frame(page1), width=30)
btn1.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
btn1.pack(pady=10)

btn2 = tk.Button(sidebar, text="Prediction", command=lambda: show_frame(page2), width=30)
btn2.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
btn2.pack(pady=10)

btn3 = tk.Button(sidebar, text="About Foot ulcer", command=lambda: show_frame(page3), width=30)
btn3.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
btn3.pack(pady=10)

btn4 = tk.Button(sidebar, text="Send Email", command=lambda: show_frame(page4), width=30)
btn4.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
btn4.pack(pady=10)

btn5 = tk.Button(sidebar, text="Contact us", command=lambda: show_frame(page5), width=30)
btn5.configure(background='blue', foreground='white', font=('arial', 10, 'bold'))
btn5.pack(pady=10)

# Display the first page by default

show_frame(page1)

# Run the main loop
root.mainloop()
