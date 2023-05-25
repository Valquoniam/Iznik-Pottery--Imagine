import os
import csv
import shutil

def move_good_images():
    # Chemin vers le fichier CSV
    csv_file = "iznik_labels.csv"
    # Chemin du dossier contenant les images
    dossier_images = "Iznik_pottery_tiles/"

    # Chemin du dossier de destination pour les images avec le label '1'
    dossier_destination = "Iznik_tiles/"

    # Ouvrir le fichier CSV
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        # Parcourir les lignes du fichier CSV
        for ligne in csv_reader:
            # Extraire le nom de l'image et le label de chaque ligne
            nom_image, label = ligne
            # Si le label est '1', déplacer l'image vers le dossier de destination
            if label == ' 1':
                print('ok')
                # Construire le chemin complet de l'image source
                chemin_image_source = os.path.join(dossier_images, nom_image)
                # Construire le chemin complet de l'image de destination
                chemin_image_destination = os.path.join(dossier_destination, nom_image)
                # Déplacer l'image vers le dossier de destination
                shutil.move(chemin_image_source, chemin_image_destination)