import tkinter as tk
import joblib
from tkinter import messagebox


model = joblib.load("model.pkl")

def label_to_str(x):
    if x == 0:
        return 'Negatif'
    else:
        return 'Positif'


def btn_event():
    text = editText.get()
    print()
    h = model.predict([text])
    hasil = label_to_str(h[0])
    tk.messagebox.showinfo("Hasil", "Kalimat yang anda masukkan besentimen : "+hasil)


form = tk.Tk()
form.title("ydhnwb")

label1 = tk.Label(form, text = "Masukkan sebuah kalimat (ENG)")
editText = tk.Entry(form)
btn = tk.Button(form, text = "Submit", bg = "purple", fg = "white", command = lambda : btn_event())

label1.grid(row = 0)
editText.grid(row = 1)
btn.grid(row = 2)

form.mainloop()

