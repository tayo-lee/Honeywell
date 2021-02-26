#import tkinter for GUI
import tkinter as tk
from tkinter import ttk
from tkinter import * # I added this line
from tkinter import messagebox # I added this line
#font types
LARGE_FONT = ("Verdana", 12)
top=Tk() # I added this line
top.geometry("100x100") #I added this line


#import stuff for graph
import matplotlib
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

#import animation to make graph live
import matplotlib.animation as animation
from matplotlib import style
style.use("seaborn-darkgrid")
#animate function
f = Figure(figsize=(20,20), dpi=100)
plot1 = f.add_subplot(211)
plot2 = f.add_subplot(212)
def animate(ii):
    pullData = open("recordfile_img.csv","r").read()
    dataList = pullData.split('\n')
    tList = []
    MList = []
    M1List = []
    for eachLine in dataList:
        if len(eachLine) >1:
            Number, Date, Time, Label,Mask_count,No_Mask_Count,Violation_count = eachLine.split(',')
            tList.append((Time))
            MList.append((Mask_count))
            M1List.append((No_Mask_Count))
            #No_Mask_Count=No_Mask_Count.replace(',','')
            #No_Mask_Count=No_Mask_Count.replace(' ','')
            if No_Mask_Count=="6": # I added this line
                messagebox.showwarning("warning"," Camera needs attention too many violations!") # I added this line
                
            

    #plot graphs
    plot1.clear()
    plot1.plot(tList, MList, 'r')
    plot2.clear()
    plot2.plot(tList, M1List, 'b')
    #add labels and config axis
    plot1.set_title("Mask Detection Results ")
    plot1.set_autoscaley_on(True) # I have changed this to True
    plot1.set_autoscalex_on(True) # I have changed this to True
    plot1.set_ylabel("Count of People With Masks (v)")
    #plot1.set_xlim(t-60,t)
    #plot1.axes.set_ybound(0,1)
    #plot1.set_ylim(2,4)
    #plot1.set_xlim(500, 510)

    plot2.set_ylabel("Count of People without Masks (v)")
    plot2.set_xlabel("Time (s)")
    #plot2.set_ybound(0,5)



#initialization
class AllWindow(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #add title
        tk.Tk.wm_title(self, "TEAM RAVIOLI COVID TRACKER")

        container = tk.Frame(self)      
        container.pack(side="top", fill="both", expand=True)      
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)      
        #show the frames
        self.frames = {}
        for F in (HomePage, ControlPanel, Settings):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(HomePage)

    def show_frame(self, cont):       
        frame = self.frames[cont]
        frame.tkraise()


#add home page
class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Dashboard", font = LARGE_FONT)
        label.pack(pady=10, padx=10)

        #quit button
        quitButton = tk.Button(self, text="QUIT", fg='red',
                                command=quit)
        quitButton.pack()
        #navigation button
        navibutton1 = ttk.Button(self, text="Control Panel",
                            command=lambda: controller.show_frame(ControlPanel))
        navibutton1.pack()
        navibutton2 = ttk.Button(self, text="Settings",
                            command=lambda: controller.show_frame(Settings))
        navibutton2.pack()

        #add graph to dashboard

        #bring up canvas
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand = True)
        #add navigation bar
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand = True)


#add control panel page
class ControlPanel(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Control Panel", font = LARGE_FONT)
        label.pack(pady=10, padx=10)

        #control buttons
        self.lightButton = tk.Button(self,
                             text="Light OFF",
                             bg= "red",
                             command=self.toggle_light)
        self.lightButton.pack()

        #navigation button
        navibutton1 = ttk.Button(self, text="Back to Dashboard",
                            command=lambda: controller.show_frame(HomePage))
        navibutton1.pack()
        navibutton2 = ttk.Button(self, text="Settings",
                            command=lambda: controller.show_frame(Settings))
        navibutton2.pack()

        #fcns triggered by control button
        #fcn to turn LED on or off
    def toggle_light(self):
        if LED1.value == 0:
            LED1.value = 1
        #change light button color
            self.lightButton.configure(bg= "green")
            self.lightButton.configure(fg= "white")
            self.lightButton.configure(text = "Light ON")
        else:
            LED1.value = 0
        #change light button color to red if light off
            self.lightButton.configure(bg= "red")
            self.lightButton.configure(text = "Light OFF")
#add settings page
class Settings(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Settings", font = LARGE_FONT)
        label.pack(pady=10, padx=10)
        #navigation button
        navibutton1 = ttk.Button(self, text="Back to Dashboard",
                            command=lambda: controller.show_frame(HomePage))
        navibutton1.pack()
        navibutton2 = ttk.Button(self, text="Go to Control Panel",
                            command=lambda: controller.show_frame(ControlPanel))
        navibutton2.pack()

app = AllWindow()
app.geometry('1025x690')
#update animation first
ani = animation.FuncAnimation(f, animate, interval=1000)
top.mainloop() # I added this line
app.mainloop()
