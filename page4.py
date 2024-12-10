import tkinter as tk
import tkinter as tk
import os
from tkinter import messagebox
from tkinter import filedialog, messagebox
from tkinter import filedialog, messagebox

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

SENDER_EMAIL = "ourprojectemails@gmail.com"
SENDER_PASSWORD = "oxipcucyayarblht"

# SMTP Server details
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
class Page4(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.background_image = tk.PhotoImage(file="bg (9).png")  # Replace with your image path
        # Create a label to display the background image
        background_label = tk.Label(self, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        tk.Label(self, text="Recipient Email:", font=("Arial", 12)).pack(pady=5)
        self.recipient_entry = tk.Entry(self, width=40)
        self.recipient_entry.pack(pady=5)

        # Subject
        tk.Label(self, text="Subject:", font=("Arial", 12)).pack(pady=5)
        self.subject_entry = tk.Entry(self, width=40)
        self.subject_entry.pack(pady=5)

        # Message Body
        tk.Label(self, text="Message:", font=("Arial", 12)).pack(pady=5)
        self.message_text = tk.Text(self, width=40, height=10)
        self.message_text.pack(pady=5)

        # Send Button
        send_button = tk.Button(self, text="Send Email", command=self.send_email)
        send_button.pack(pady=20)

    def send_email(self):
        recipient = self.recipient_entry.get()
        subject = self.subject_entry.get()
        message_body = self.message_text.get("1.0", tk.END)

        if not recipient or not subject or not message_body.strip():
            messagebox.showwarning("Input Error", "All fields are required!")
            return

        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL


            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(message_body, 'plain'))

            # Send the email
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, recipient, msg.as_string())

            messagebox.showinfo("Success", "Email sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send email: {e}")
