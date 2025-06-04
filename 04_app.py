from tkinter import Tk, filedialog, Label, Button
from PIL import ImageTk, Image
from inference import classify_image


def open_file():
    filepath = filedialog.askopenfilename(
        title="Select an image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not filepath:
        return

    label.config(text="Processing...")
    root.update()

    try:
        label_img = Image.open(filepath)
        label_img.thumbnail((1080, 1080))
        img = ImageTk.PhotoImage(label_img)
        panel.config(image=img)
        panel.image = img

        label_text, conf = classify_image(filepath)
        label.config(text=f"Prediction: {label_text} ({conf:.2f})")
    except Exception as e:
        label.config(text=f"Error: {e}")


root = Tk()
root.title("Eve-Teasing Image Classifier")

panel = Label(root)
panel.pack()

label = Label(root, text="Choose an image to classify", font=("Arial", 14))
label.pack()

btn = Button(root, text="Browse", command=open_file)
btn.pack()

root.mainloop()
