import tkinter
from tkinter import *
from tkinter import Label, PhotoImage
import joblib


lour = tkinter.Tk()
lour.geometry("900x650")
lour.title('SMS SPAM')
bg = PhotoImage(file ='22.png')#Enter your picutre path that is already in the project folder.
label1 = Label(lour, image = bg)
label1.place(x = 0, y = 0)

entry_font = ("Comic Sans MS", 40)
loaded_model = joblib.load('finalized_model.sav')

comic_sans_font = ("Comic Sans MS", 20)
comic_sans_font1 = ("Comic Sans MS", 17)
comic_sans_font2 = ("Comic Sans MS", 20)

def submit_text():
    entered_text = [str(entry.get())]
    loaded_vectorizer = joblib.load('vectorizer.joblib')
    new_data_transformed = loaded_vectorizer.transform(entered_text)
    predictions1 = loaded_model.predict(new_data_transformed)
    if predictions1[0]=="ham":
         result_label.config(text=f"(Your Text is: HAM \U0001F389 😊 ) ")
    else:
        result_label.config(text=f"(Your Text is: SPAM \U00002620 😞 )")

label = tkinter.Label(lour, text="Enter your text here! :",bd=3, font=comic_sans_font2)
label.grid( column=1, padx= 100, pady= 30)

entry = tkinter.Entry(lour, width=50, bd=3, font=comic_sans_font2)
entry.grid( column=1, padx= 40, pady= 20)
#entry.pack(pady=20)

button = tkinter.Button(lour, text='Check the SMS if it is SPAM or NOT', width=30,bd=3, command=submit_text,font=comic_sans_font1,bg="pink",anchor="w")
#button.pack(pady=30)
button.grid( column=1, padx= 40, pady= 20)

result_label = tkinter.Label(lour,text="",fg="white",bg="orange",font=comic_sans_font)
#result_label.pack()
result_label.grid( column=1, padx= 40, pady= 20)

lour.mainloop()
#Check the SMS if it is SPAM or NOT

