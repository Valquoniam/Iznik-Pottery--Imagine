from flask import Flask, render_template, send_file, request, redirect, url_for
import os
import sys
sys.path.append('../')
from util.results import Results
import argparse
import atexit
import webbrowser
import subprocess
import shutil
import util.results
import platform
from time import sleep

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Récupérer le nom du fichier source d'images en tant qu'argument
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Fichier source d\'images')
args = parser.parse_args()
image_source_file = args.file

results = Results(image_source_file)
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/')
def index2():
    return render_template('index.html')


@app.route('/gallery.html', methods=['GET','POST'])
def display_images():
    results.copy_files('static/images')
    
    # Récupérer la liste des noms de fichiers d'images dans le dossier
    image_files = [f for f in os.listdir('static/images') if os.path.isfile(os.path.join('static/images', f)) and f != '.gitkeep' and f != 'style.css' and f != 'script.js' and f!="grid.jpg" and f!="icon.ico"]
    
     # Check if the request is a refresh
    if request.method == 'GET' and 'HTTP_CACHE_CONTROL' in request.environ and request.environ['HTTP_CACHE_CONTROL'] == 'max-age=0':
        #clear app cache
        app.jinja_env.cache = {}
        return render_template('/gallery.html', image_names=image_files)

    if request.method == 'POST':
        if str(request.form) == "ImmutableMultiDict([('generate', 'Generate new images')])":
            subprocess.run(['python', '../main_tools/generate.py', '--model', '../training_results/iznik_snapshot.pkl', '--output-dir', '../results/images', '--truncation-psi', '0.9'])
            # refresh the page
            results.copy_files('static/images')
            # Récupérer la liste des noms de fichiers d'images dans le dossier
            image_files = [f for f in os.listdir('static') if os.path.isfile(os.path.join('static', f)) and f != '.gitkeep' and f != 'style.css' and f != 'script.js' and f!="grid.jpg" and f!="icon.ico"]
            return redirect(url_for('display_images'))

        if len(request.form) == 3:
            grid_width=int(request.form['grid-height'])
            grid_height=int(request.form['grid-width'])
            
            images = results.get_images(image_source_file)

            results.save_grid(images, (grid_width, grid_height),'../grids/grid.jpg')
            return send_file('../results/grids/grid.jpg', as_attachment=True)

    # Rendre le template HTML et passer les chemins des images à afficher
    
    return render_template('gallery.html', image_names=image_files)

@app.route('/videos.html', methods=['GET','POST'])
def view_videos():
    # Copy videos to static folder
    videos_dir = '../results/videos'
    static_dir = 'static/videos'
    if os.path.exists(static_dir):
        shutil.rmtree(static_dir)
    shutil.copytree(videos_dir, static_dir)
    
    # Render page with videos
    videos = os.listdir(static_dir)
    return render_template('videos.html', videos=videos)

def delete_images():
    folder = 'static'
    for root, _ , files in os.walk(folder):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                if os.path.isfile(file_path) and (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.gif') or filename.endswith('.mp4')):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')

@app.route('/tensorboard')
def show_tb():
    subprocess.run(["screen", "-dmS", "tensorboardgen", "tensorboard", "--logdir", "../training_results", "--host", "127.0.0.1", "--port", "3025"])
    if platform.system() == 'Windows':
        # Classic Windows
        url = 'http://127.0.0.1:3025/'
        
    elif platform.system() == 'Linux' and ("microsoft" in platform.uname().release.lower() and "microsoft" in platform.uname().version.lower()):
        # WSL2
        url = 'http://172.27.201.41:3025/'
    else:
        # Classic Linux
        url = 'http://127.0.0.1:3025/'
    sleep(5)
    webbrowser.open(url)
    
    return render_template("index.html")

# Register the function to be called when the program exits
atexit.register(delete_images)

if __name__ == '__main__':
    
    print(platform.system())
    
    if platform.system() == 'Windows':
        # Classic Windows
        url = 'http://127.0.0.1:5000/'
        
    elif platform.system() == 'Linux' and ("microsoft" in platform.uname().release.lower()):
        # WSL2
        print("oue wsl")
        url = 'http://172.27.201.41:5000/'
    else:
        # Classic Linux
        url = 'http://127.0.0.1:5000/'
    webbrowser.open(url)
    app.run(host="0.0.0.0")
    

