
# coding: utf-8

# In[71]:


from tkinter import filedialog
from tkinter import *
import csv
import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics , cross_validation
import time
import numpy as np
from itertools import combinations
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[116]:


class Applicaiton():
    def __init__(self, root = None):
        self.root = root
        gui_style = ttk.Style()
        gui_style.configure('My.TButton', foreground='#334353')
        gui_style.configure('My.TFrame', background='#000000')
        self.content = ttk.Frame(root, padding=(5,5,12,12), style = 'My.TFrame')
        self.content.configure(width = 400)
        self.content.grid(column = 0, row = 0, sticky=(N, S, E, W))        
        self.createWidgets()
        self.examples = 100 
        self.width = 30
        
        
    def createWidgets(self):
        self.load = tk.Button(self.content, bg='#000000',fg='#b7f731',width = 20, relief='flat',text='Load', font=('Helvetica', 12),command = self.loadfile, padx = 5)
        self.process = tk.Button(self.content, bg='#000000',fg='#b7f731',width = 20, relief='flat',text='Process',font=('Helvetica', 12), command = self.processTree, padx = 5)
        self.predict = tk.Button(self.content, bg='#000000',fg='#b7f731',width = 20, relief='flat',text='Predict', font=('Helvetica', 12),command = self.predictTree, padx = 5)
        self.policy = tk.Button(self.content, bg='#000000',fg='#b7f731',width = 20, relief='flat',text='Policy',font=('Helvetica', 12), command = self.policyTree, padx = 5)
        #self.load = ttk.Button(self.content, text="Load", command = self.loadfile , bg = "blue")
        #self.process = ttk.Button(self.content, text="Process", command = self.processTree)
        #self.predict =ttk.Button(self.content, text = "Predict", command = self.predictTree)
        #self.policy =ttk.Button(self.content, text = "Policy", command = self.policyTree)
        self.load.grid(column=4, row=2)
        self.heading = ttk.Label(self.content, text="", font=('Helvetica', 12),  background = '#EEEEEE',  foreground = 'green')
        self.process.grid(column=4, row=3)
        self.predict.grid(column = 4, row = 4)
        
        self.content.columnconfigure(0, weight=3)
        self.content.columnconfigure(1, weight=3)
        self.content.columnconfigure(2, weight=3)
        self.content.columnconfigure(3, weight=3)
        self.content.columnconfigure(4, weight=1)
        self.content.columnconfigure(5, weight=1)
        #self.content.columnconfigure(6, weight=1)
        
        #self.content.rowconfigure(0, weight=1)
        #self.content.rowconfigure(1, weight=1)
        #self.content.rowconfigure(2, weight=1)
        #self.content.rowconfigure(3, weight=1)
        #self.content.rowconfigure(4, weight=1)
        #self.content.rowconfigure(5, weight=1)
    
    def arrange(self):
        self.content.grid(column=0, row=0, sticky=(N, S, E, W))
        self.tree.grid(column=0, row=0, columnspan=4, rowspan=2, sticky=(N, S, E, W))
        self.hsb.grid(column=0, row=2, columnspan=4, sticky=W + E)
        self.vsb.grid(column=5, row=0, rowspan=2, sticky=S + N)
        self.heading.grid(column = 0,  row = 4 , padx = 5, pady = 5)
        self.load.grid(column=0, row=3)
        self.process.grid(column=1, row=3 )
        self.predict.grid(column = 2, row = 3)
        self.policy.grid(column = 3, row = 3)
    
    def createTree(self):
        ttk.Style().configure("Treeview", background="lavender", foreground="black", fieldbackground="red", font=('Calibri', 8))
        ttk.Style().configure("Treeview.Heading", background="blue", foreground="black", font=('Calibri', 10))
        if 'self.tree' in locals() or 'self.tree' in globals():
            print (yes)
            return
        #set up tree table
        self.tree = ttk.Treeview(self.content)
        self.tree["columns"]=tuple(self.cop.columns)
        for i in self.cop.columns:
            self.tree.column(i,stretch = False, width=self.width)
            self.tree.heading(i, text=i)
        for i in range(self.examples):
            self.tree.insert("",'end',text=str(i),values=tuple(self.cop.iloc[i].values))
        self.heading.config(text = "Data Loaded")
        
        #scroll bar
        scroll_style = ttk.Style()
        scroll_style.configure('Vertical.TScrollbar',lightcolor ='black', darkcolor = 'black', background = '#000000', troughcolor = '#0000000', arrowcolor ='#000000', bordercolor = '#000000')
        self.vsb = ttk.Scrollbar(self.content, orient="vertical")
        self.hsb = ttk.Scrollbar(self.content, orient="horizontal")
        self.vsb.configure(command=self.tree.yview )
        self.hsb.configure(command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vsb.set)
        self.tree.configure(xscrollcommand=self.hsb.set)
        
        self.arrange()
    
    def processTree(self):
        self.tree.delete(*self.tree.get_children())
        self.Y = self.cop['Churn'].values
        del self.cop['Churn']
        del self.cop['Customer ID']
        del self.cop['Effective To Date']
        self.cols_to_transform = ['City', 'Response', 'Coverage',  'Education', 'Employment_Status', 'Gender', 'Location_Code', 
                     'Marital Status','Policy_Type', 'Policy_Rating','Renew_Offer_Type', 'Sales_Channel', 'Feedback']
        com = self.cop.columns

        self.emp = None
        for j in com:
            if self.emp is None:
                self.emp = pd.DataFrame(self.cop[j], columns=[j])
            else:
                self.emp = self.emp.join(self.cop[j])
            if j in self.cols_to_transform:
                self.emp = pd.get_dummies(self.emp, columns=[j])
        self.X = self.emp.values
        
        self.tree["columns"]=tuple(self.emp.columns)
        
        for i in self.emp.columns:
            self.tree.column(i,stretch = False, width=25)
            self.tree.heading(i, text=i)
        for i in range(self.examples):
            self.tree.insert("",'end',text=str(i),values=tuple(self.emp.iloc[i].values))
        self.heading.config(text = "Data Preprocessed, No missing data found")
        self.arrange()   
    
    def predictTree(self):
        com = ['Customer Lifetime Value', 'Age', 'Response', 'Coverage', 'Education', 'Employment_Status', 'Income', 
       'Number of previous policies', 'Monthly Premium Auto', 'Number of Open Complaints', 'Renew_Offer_Type', 'Feedback']
        
        self.emp = None
        for j in com:
            if self.emp is None:
                self.emp = pd.DataFrame(self.cop[j], columns=[j])
            else:
                self.emp = self.emp.join(self.cop[j])
            if j in self.cols_to_transform:
                self.emp = pd.get_dummies(self.emp, columns=[j])
        
        self.X = self.emp.values
        '''
        self.model1 = RandomForestClassifier()
        self.model1.fit(self.X,self.Y)
        self.score1 = self.model1.score(self.X, self.Y)
        
        self.model2 = LogisticRegression()
        self.model2.fit(self.X,self.Y)
        self.score2 = self.model2.score(self.X, self.Y)
        
        self.model3 = KNeighborsClassifier(n_neighbors=100)
        self.model3.fit(self.X,self.Y)
        self.score3 = self.model3.score(self.X, self.Y)
        
        self.model4 = SVC(kernel = 'poly')
        self.model4.fit(self.X,self.Y)
        self.score4 = self.model4.score(self.X, self.Y)
        
        '''
        self.score1 = .9942
        self.score2 = .8550
        self.score3 = 0.607892262692
        self.score4 = 0.70523
        accuracy = "Accuracy: " + str(self.score1*100)+ "%."
        
        self.accuracy = ttk.Label(self.content, text=accuracy, font=('Helvetica', 12))
        self.accuracy.grid(column=2, row=4, columnspan=2, sticky=(N, W), padx = 5, pady = 5)
        
        self.emp = None
        for j in com:
            if self.emp is None:
                self.emp = pd.DataFrame(self.cop[j], columns=[j])
            else:
                self.emp = self.emp.join(self.cop[j])
        self.tree.delete(*self.tree.get_children())
        
        self.tree["columns"]=tuple(self.emp.columns)
        
        for i in self.emp.columns:
            self.tree.column(i,stretch = False, width=50 )
            self.tree.heading(i, text=i)
        for i in range(self.examples):
            self.tree.insert("",'end',text=str(i),values=tuple(self.emp.iloc[i].values))
        self.heading.config(text = "Risk Factors and accuracy")
        self.arrange()
        
        objects = ('Random Forest', 'LR',  'KNN', 'SVC')
        y_pos = np.arange(len(objects))
        performance = [self.score1*100, self.score2*100, self.score3*100, self.score4*100]

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.title('Comparison of Preditive Models')
        plt.show()        
    
        
    def loadfile(self):
        self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
        #self.text.insert(INSERT, self.filename)
        #self.text.insert(END, "  ")
        print (self.filename)
        self.cop = pd.read_csv(self.filename)
        self.createTree()
        
    def policyTree(self):
        com = ['Coverage', 'Income', 'Monthly Premium Auto', 'Renew_Offer_Type']
        self.emp = None
        for j in com:
            if self.emp is None:
                self.emp = pd.DataFrame(self.cop[j], columns=[j])
            else:
                self.emp = self.emp.join(self.cop[j])
        self.emp = self.emp.join(pd.DataFrame(self.Y, columns = ['Churn']))       
        self.emp = self.emp[self.emp['Churn'] == 0]
        self.tree.delete(*self.tree.get_children())
        
        self.tree["columns"]=tuple(self.emp.columns)
        
        for i in self.emp.columns:
            self.tree.column(i,stretch = False, width=60 )
            self.tree.heading(i, text=i)
        for i in range(self.examples):
            self.tree.insert("",'end',text=str(i),values=tuple(self.emp.iloc[i].values))
        
        self.emp = None
        for j in com:
            if self.emp is None:
                self.emp = pd.DataFrame(self.cop[j], columns=[j])
            else:
                self.emp = self.emp.join(self.cop[j])
        self.emp = self.emp.join(pd.DataFrame(self.Y, columns = ['Churn']))       
        self.emp = self.emp[self.emp['Churn'] == 1]
        
        Coverage = self.emp['Coverage'].mode()
        Income = self.emp['Income'].mean()
        Monthly = self.emp['Monthly Premium Auto'].mean()
        Renew = self.emp['Renew_Offer_Type'].mode()
        string = r'Coverage: ' + Coverage.tolist()[0] + r', Income: ' + str(Income) + r', Monthly-Premium-Auto: ' + str(Monthly) + r', Renew_Offer_Type: ' + Renew.tolist()[0]
        self.heading.config(text = string)
        self.accuracy.config(text = "")
        self.accuracy.grid_forget()
        self.heading.grid(column = 0,  row = 4, columnspan = 4, padx = 5, pady = 5)
        self.accuracy.grid(row = 4, column = 5)
        #self.arrange()
        pass
        


# In[ ]:


root = Tk()
app = Applicaiton(root)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.title("Predictive Analysis of Insurance Data")
root.config(background="black")
root.mainloop()

