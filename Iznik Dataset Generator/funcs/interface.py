from PIL import Image
import os
from tkinter import *
from PIL import Image, ImageTk

global_var_E = None

################################## PART 1 : INTERFACE AND SELECTION ################################            

"""
Main function to deal with the window
"""
    
def display_and_label(E,dataset_path):        
    
    #Initialise global variables (in order for the clicks to have an effect)
    global images_displayed
    global images_labels
    global result
    global root
    result = None
    
    global global_var_E
    global_var_E = E
    
    # Create a window
    root = Tk()
    root.geometry("600x650")
    root.title("Images Validation")

    # List of the images we will display
    image_list = []
    
    #List of the images labels
    images_labels = [1] * len(E)
    
    for image in E:
            
            # Image under tkinker format
            img = Image.open(os.path.join(dataset_path, image))
            img = img.resize((100, 100))
            photo = ImageTk.PhotoImage(img)
            image_list.append(photo)
            
    # Display images and allow to click on them
    images_displayed = []
    
    for index in range(len(image_list)):
        label = Label(root, image=image_list[index])        
        label.grid(row=int(index/5), column=index % 5, padx=7, pady=7, sticky="nsew")
        label.configure(bg= "green")
        
        # Event 'click on an image'
        label.bind("<Button-1>", image_click_label)
        
        # Memorise the image position
        images_displayed.append(label)

    # Button 'Validate'
    button = Button(root, text="Validate", command=valider_click)
    button.grid(row=6, column=1, pady=20, padx = 10)
    
    # Button 'Unvalidate everything'
    button = Button(root, text="Unvalidate all", command=_rien_valider_click)
    button.grid(row=6, column=3, pady=20, padx = 10)

    # What happens when we close the window
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Launch the mainloop
    root.mainloop()

    return result
"""
Auxiliary function do define a click on an image
"""

# Function to define what happens when we click on an image
def image_click_label(event):
    
    # récupérer le widget (label) sur lequel on a cliqué
    label = event.widget
    
    # récupérer l'index de l'image dans la liste
    index = images_displayed.index(label)
    
    # Si on a déjà cliqué sur l'image, un clic veut dire qu'on annule la sélection
    if images_labels[index] == 0:
        
        label.configure(bg= "green")
        images_labels[index] = 1
    
    # Sinon, on invalide l'image    
    else:
        
        label.configure(bg="red")
        images_labels[index] = 0
        
def image_click(label):
    index = images_displayed.index(label)
    if images_labels[index] == 0:
        label.configure(bg="green")
        images_labels[index] = 1
    else:
        label.configure(bg="red")
        images_labels[index] = 0

"""
Auxiliary function do define what a validation is
"""

def _rien_valider_click():
    for label in images_displayed:
        image_click(label)


def valider_click():
    
    E = global_var_E
    global result 
    # Noter les labels dans le fichier texte
    for index in range(len(E)): 
        write_to_file('iznik_labels.csv', f"{E[index]}, {images_labels[index]}")
        
    sort_file('iznik_labels.csv')
    
    result = images_labels
    
    # Fermer la fenêtre Tkinter
    root.destroy()

"""
Auxiliary function to close the window
"""

def on_window_close():
      
    root.destroy()
    exit('Window forcefully closed')
    
################################## PART 2 : WRITE THE LABEL IN THE TEXT FILE ################################

### Sort the csv file
def sort_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()  # Lit toutes les lignes dans une liste

    sorted_lines = sorted(lines, key=lambda line: int(line.split('.')[0].split('_')[1]))  # Trie les lignes selon le numéro extrait
     
    with open(filename, 'w') as file:
        file.writelines(sorted_lines)  # Écrit les lignes triées dans le fichier
        

"""
Auxiliary function to write in a .txt file at the desired line
"""

def write_to_file(filename, message):
     with open(filename, 'a') as file:
        file.write(message + '\n')
        
        