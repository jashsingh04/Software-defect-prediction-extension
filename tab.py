import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from algo import ACCURACY, PRECISION, RECALL, F1, AUC ,avg_accuracy, PATH, ALGORITHM
from selector import CHECK_BOX
import tkinter as tk
from tkinter import ttk
import pandas as pd

avg_precision = sum(PRECISION)/len(PRECISION)
avg_recall = sum(RECALL)/len(RECALL)
avg_f1 = sum(F1)/len(F1)
avg_auc = sum(AUC)/len(AUC)
print(avg_auc, AUC)
ACCURACY.append(avg_accuracy)
PRECISION.append(avg_precision)
RECALL.append(avg_recall)
F1.append(avg_f1)
AUC.append(avg_auc)

output_loc = ''

rows = {'Algorithm': ['Random\nForest', ' Gaussian NB', ' Adaboost', ' MLP', ' KNN', ' CNN'],
        'Accuracy': ACCURACY,
        'Precision': PRECISION,
        'Recall': RECALL,
        'F1': F1,
        'AUC': AUC}

selected_keys = ['Algorithm']
for key in rows.keys():
    if key != 'Algorithm' and CHECK_BOX[key.lower()] > 0:
        selected_keys.append(key)


selected_rows = {key: rows[key] for key in selected_keys}


max_index = ACCURACY.index(max(ACCURACY))


df = pd.DataFrame(selected_rows)

name = '( '
for _ in selected_rows:
    name += _+' '

name += ' )'
class Table(tk.Frame):
    global max_index
    def __init__(self, parent=None, dataframe=pd.DataFrame(), filename="", accuracy_cnn=""):
        super().__init__(parent)
        self.filename_label = tk.Label(self, text=f"FILE NAME: {PATH.split('/')[-1]}", font=('Arial', 12))
        self.filename_label.pack(side='top', pady=10)

        algo = f"Algorithm : {ALGORITHM}"

        self.algo = tk.Label(self, text=f"Algorithm: {ALGORITHM}", font=('Arial', 12))
        self.algo.pack(side='top', pady=10)


        self.table = ttk.Treeview(self, columns=list(dataframe.columns), show='headings', style='Custom.Treeview')
        self.table.pack(side='top', fill='both', expand=True, padx=20, pady=20)

        for col in self.table["columns"]:
            self.table.heading(col, text=col)

        dataframe_rows = dataframe.to_numpy().tolist()
        for row in dataframe_rows:
            self.table.insert('', 'end', values=row)

        max_accuracy_algo = dataframe.loc[max_index, 'Algorithm']
        max_accuracy_text = f"Maximum accuracy is achieved by \n\n{max_accuracy_algo}"


        self.max_accuracy_label = tk.Label(self, text=max_accuracy_text, font=('Arial', 12))
        self.max_accuracy_label.pack(side='top', pady=10)



        self.ok_button = ttk.Button(self, text="OK", command=self.save_and_close)
        self.ok_button.pack(side='top', pady=10)

        # Set weight of the Frame's rows to make the table stay in the center
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # Set weight of the Frame's columns to make the table expand to fill the available space
        self.columnconfigure(0, weight=1)

        # Center the widgets horizontally
        self.filename_label.pack(side='top', fill='x', padx=10)
        self.max_accuracy_label.pack(side='top', fill='x', padx=10)
        self.ok_button.pack(side='top', pady=10)

        # Set weight of the table's columns to make them expand to fill the available space
        for i, col in enumerate(dataframe.columns):
            self.table.column(col, width=100, minwidth=100, stretch=True)

    def show_chart(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        num_colors = len(colors)
        cols = [col for col in df.columns[1:] if col != 'Accuracy']
        if 'Accuracy' in df.columns:
            cols.append('Accuracy')
        for i, col in enumerate(cols):
            ax.plot(df['Algorithm'], df[col], 'o-', label=col, color=colors[i % num_colors])
        ax.set_title('Performance Metrics per Algorithm')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right')

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def save_and_close(self):
        # Save the table data to a CSV file
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        num_colors = len(colors)
        cols = [col for col in df.columns[1:] if col != 'Accuracy']
        if 'Accuracy' in df.columns:
            cols.append('Accuracy')
        for i, col in enumerate(cols):
            ax.plot(df['Algorithm'], df[col], 'o-', label=col, color=colors[i % num_colors])
        ax.set_title('Performance Metrics per Algorithm')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right')

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        filename = f"{PATH.split('/')[-1][:-4]} with {ALGORITHM} {name} results.csv"
        df.to_csv(f'{output_loc}{ALGORITHM}/{filename}', index=False)
        fig.savefig(f'{output_loc}{ALGORITHM}/{PATH.split("/")[-1][:-4]}_{name}.png')

        # Close the GUI
        self.master.destroy()


root = tk.Tk()
root.title('Results')
table = Table(root, dataframe=df)
table.show_chart()

table.pack(side='top', fill='both', expand=True)
root.mainloop()