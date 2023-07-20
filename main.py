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

def run_process(args):
    if args[1] == "web_display.py":
        os.chdir("web_display")
    if args[1] == "generate.py":
        os.chdir("main_tools")
    try:
        subprocess.run(args)
    except KeyboardInterrupt:
        pass


def main():

    if not os.path.exists("training_results/iznik_snapshot.pkl"):
        subprocess.run(["gdown", "1JaogrbRCWgNDy4MBbZu-SD1j_3ZTe6uY", "-O", "training_results/iznik_snapshot.pkl"])

    if not os.path.exists("results"):
        os.makedirs("results")
        os.makedirs("results/images")
        os.makedirs("results/grids")
        os.makedirs("results/videos")
        
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

    # Web display
    parser.add_argument("--web_display", "-w", help="Display images on web", default=None, action="store_true")

    # Show tensorboard
    parser.add_argument("--tensorboard", "-tb", help="Show tensorboard", default=None, action="store_true")
    args = parser.parse_args()

    pool = multiprocessing.Pool(processes=3)
    if args.tensorboard:
        try:
            pool.apply_async(run_process, [["tensorboard", "--logdir", "training_results", "--host", "127.0.0.1", "--port", "3025"]])
            time.sleep(3)
            webbrowser.open("http://127.0.0.1:3025/")
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

    try:
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()


if __name__ == "__main__":
    main()