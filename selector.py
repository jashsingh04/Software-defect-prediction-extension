import tkinter as tk
from tkinter import ttk, filedialog, IntVar

ALGORITHM, PATH = '', ''
CHECK_BOX = {}
class GUI:

    def __init__(self, master):
        self.master = master
        master.title("File Selector")
        self.master.update()
        self.master.geometry(f"450x290")  # Resize the window
        self.master.attributes('-alpha', 0.6)



        self.watermark_label = ttk.Label(master, text="Copyright © 2023.\n     All rights reserved.", font=("Arial", 10), foreground="gray")
        self.watermark_label.pack(side="bottom", anchor="se", padx=10, pady=10)
        # Place the label in the center of the window


        # Create the style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Arial", 12), foreground="black")
        style.configure("TButton", font=("Arial", 12), padding=5, foreground="black")
        master.attributes('-alpha', 0.9)



        # Create the widgets
        self.file_label = ttk.Label(master, text="No file selected.")
        self.file_label.pack(pady=10)

        self.file_type = tk.StringVar()
        self.file_type.set("Select Algorithm")
        self.file_menu = ttk.OptionMenu(master, self.file_type, *["BAT", "Sparrow Search", "BAT", "Squirrel Search"], command=self.update_file_type)
        self.file_menu.pack(pady=10)

        # Create a frame to contain the check buttons
        self.check_frame = ttk.Frame(master)
        self.check_frame.pack(pady=5)

        self.accuracy_var = IntVar(value=1)
        self.accuracy_checkbox = ttk.Checkbutton(self.check_frame, text="Accuracy", variable=self.accuracy_var)
        self.accuracy_checkbox.pack(side="left", padx=5, anchor="w")  # Left align the button

        self.recall_var = IntVar()
        self.recall_checkbox = ttk.Checkbutton(self.check_frame, text="Recall", variable=self.recall_var)
        self.recall_checkbox.pack(side="left", padx=5, anchor="w")  # Left align the button

        self.precision_var = IntVar()
        self.precision_checkbox = ttk.Checkbutton(self.check_frame, text="Precision", variable=self.precision_var)
        self.precision_checkbox.pack(side="left", padx=5, anchor="w")  # Left align the button

        self.f1_var = IntVar()
        self.f1_checkbox = ttk.Checkbutton(self.check_frame, text="F1", variable=self.f1_var)
        self.f1_checkbox.pack(side="left", padx=5, anchor="w")  # Left align the button

        self.auc_var = IntVar()
        self.auc_checkbox = ttk.Checkbutton(self.check_frame, text="AUC", variable=self.auc_var)
        self.auc_checkbox.pack(side="left", padx=5, anchor="w")  # Left align the button

        self.file_button = ttk.Button(master, text="Select File", command=self.browse_file)
        self.file_button.pack(pady=10)

        self.submit_button = ttk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)

        # Initialize the file_path variable
        self.file_path = ""

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.file_label.config(text=file_path)
        self.file_path = file_path

    def update_file_type(self, selected_type):
        self.file_type.set(selected_type)

    def submit(self):
        global ALGORITHM, PATH, CHECK_BOX
        algorithm = self.file_type.get()

        CHECK_BOX = {
            "accuracy": self.accuracy_var.get(),
            "recall": self.recall_var.get(),
            "precision": self.precision_var.get(),
            "f1": self.f1_var.get(),
            "auc": self.auc_var.get()
        }

        PATH = self.file_path
        ALGORITHM = algorithm

        # Close the GUI
        self.master.destroy()

root = tk.Tk()
my_gui = GUI(root)
my_gui.accuracy_checkbox.pack(anchor='w') # Left align checkbox
my_gui.recall_checkbox.pack(anchor='w') # Left align checkbox
my_gui.precision_checkbox.pack(anchor='w') # Left align checkbox
my_gui.f1_checkbox.pack(anchor='w') # Left align checkbox
my_gui.auc_checkbox.pack(anchor='w') # Left align checkbox
root.mainloop()




