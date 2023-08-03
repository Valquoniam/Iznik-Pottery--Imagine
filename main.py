import argparse
import torch
import subprocess
import os
import multiprocessing
import webbrowser
import sys
import shutil
sys.path.append("..")
import time
import platform
import gdown
from util.results import convert_to_png

def run_process(args):
    if args[1] == "web_display.py":
        os.chdir("web_display")
    if args[1] == "generate.py" or args[1] == "run_network.py":
        os.chdir("main_tools")
    try:
        subprocess.run(args)
    except KeyboardInterrupt:
        pass


def main():

    if not os.path.exists("results/images"):
        os.makedirs("results/images")
        
    if not os.path.exists("training_results/iznik_snapshot.pkl"):
        subprocess.run(["gdown", "1JaogrbRCWgNDy4MBbZu-SD1j_3ZTe6uY", "-O", "training_results/iznik_snapshot.pkl"])

    if not os.path.exists("results"):
        os.makedirs("results")
        os.makedirs("results/images")
        os.makedirs("results/grids")
        os.makedirs("results/videos")
    
    if not os.path.exists("web_display/static/images"):
        os.makedirs("web_display/static/images")
        os.makedirs("web_display/static/grids")
        os.makedirs("web_display/static/videos")
        
    torch.cuda.empty_cache() #type: ignore

    # ---------------------------------- PARSER ----------------------------------
    parser = argparse.ArgumentParser(description='All TileGAN functionalities')
    gen_group = parser.add_argument_group('required arguments to generate images')
    
    # To generate images
    gen_group.add_argument("--generate", "-g", help="Generate images", default=None, action="store_true")
    gen_group.add_argument("--model", "-m", help="Filename for a snapshot to resume", default="../training_results/iznik_snapshot.pkl", type=str)
    gen_group.add_argument("--gpus", help="Comma-separated list of GPUs to be used (default: %(default)s)",
                           default="0", type=str)
    gen_group.add_argument("--output-dir", "-o", help="Root directory for experiments (default: %(default)s)",
                           default="../results/images", metavar="DIR")
    gen_group.add_argument("--images-num", "-i", help="Number of images to generate (default: %(default)s)",
                           default=32, type=int)
    gen_group.add_argument("--truncation-psi", "-t",
                           help="Truncation Psi to be used in producing sample images (default: %(default)s)",
                           default=1.3, type=float)
    gen_group.add_argument("--ratio", "-r", help="Crop ratio for output images (default: %(default)s)",
                           default=1.0, type=float)

    # To generate latent vectors
    parser.add_argument("--generate-latent", "-gl", help="Generate latent vectors", default=None, action="store_true")
    # Web display
    parser.add_argument("--web_display", "-w", help="Display images on web", default=None, action="store_true")

    # Show tensorboard
    parser.add_argument("--tensorboard", "-tb", help="Show tensorboard", default=None, action="store_true")
    parser.add_argument("--projector", "-pr", help="Show projector", default=None, action="store_true")
    
    # Download dataset
    parser.add_argument("--dataset", "-d", help="Download dataset", default=None, action="store_true")
    
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=3)
    if args.tensorboard:
        try:
            pool.apply_async(run_process, [["tensorboard", "--logdir", "training_results", "--host", "127.0.0.1", "--port", "3025"]])
            time.sleep(5)
            if platform.system() == 'Windows':
                # Classic Windows
                url = 'http://127.0.0.1:3025/'
                
            elif platform.system() == 'Linux' and ("microsoft" in platform.uname().release.lower() and "microsoft" in platform.uname().version.lower()):
                # WSL2
                url = 'http://172.27.201.41:3025/'
            else:
                # Classic Linux
                url = 'http://127.0.0.1:3025/'
            
            if args.projector:
                url = url + '?darkMode=true#projector'

            webbrowser.open(url)
            
        except KeyboardInterrupt:
            pass

    if args.generate:
        shutil.rmtree("results/images")
        os.makedirs("results/images", exist_ok=True)
        try:
            pool.apply_async(run_process, [["python", "generate.py", "--model", args.model, "--gpus", args.gpus,
                                            "--output-dir", args.output_dir, "--images-num", str(args.images_num),
                                            "--truncation-psi", str(args.truncation_psi), "--ratio", str(args.ratio), "--images-num", str(args.images_num)]])
        except KeyboardInterrupt:
            pass
    
    if args.web_display:
        try:
            pool.apply_async(run_process, [["python", "web_display.py", "-f", "../results/images"]])
        except KeyboardInterrupt:
            pass
        
    if args.generate_latent:
        try:
            pool.apply_async(run_process, [["python", "run_network.py", "--pretrained-pkl", args.model, "--gpus", args.gpus, "--vis", "--dataset", "iznik", "--vis-latents"]])
        except KeyboardInterrupt:
            pass
        #PD gros PD
    
    if args.dataset:
        try:
            if os.path.exists("iznik"):
                shutil.rmtree("iznik")
            if not os.path.exists("datasets/iznik"):
                gdown.download(" https://drive.google.com/uc?id=18DTw7eVQa1D96Nno1-uOwjBgfJl7Ihy7", "dataset.zip", quiet=False, use_cookies=False)
                subprocess.run(["unzip", "dataset.zip"])
                subprocess.run(["rm", "dataset.zip"])
                os.rename("Iznik_tiles", "iznik")
                convert_to_png("iznik")
                os.chdir("main_tools")
                subprocess.run(["python","prepare_data.py", "--task", "iznik", "--images-dir", "../iznik", "--format", "png"])
                subprocess.run(["mv", "datasets", "../datasets"])
                shutil.rmtree("../iznik")
        except KeyboardInterrupt:
            pass
    
    try:
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()