#pip install supervision
#pip install autodistill
#pip install autodistill-grounded-sam

import subprocess
import supervision as sv
import tqdm

from Matthias.detection import CaptionOntology
#from autodistill_grounded_sam import GroundedSAM

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
#base_model = GroundedSAM(ontology=autodistill.CaptionOntology({"shipping container": "container"}))

VIDEO_DIR_PATH = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/videos"
IMAGE_DIR_PATH = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/images"
DATASET_DIR_PATH = "C:/Users/SkumJustEatMe/CDIO_3_2024/image_detection/data/dataset"
FRAME_STRIDE = 10

# def install_packages():
    # packages_to_install = [
    # "autodistill",
    # "autodistill-grounded-sam",
    # "autodistill-yolov8",
    # "roboflow",
    # "supervision==0.9.0"
    # ]

    # for package in packages_to_install:
    #     subprocess.run(["pip", "install", "-q", package])

def cut_video_into_images():

    video_paths = sv.list_files_with_extensions(directory=VIDEO_DIR_PATH, extensions=["mov", "mp4"])

    TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[:2], video_paths[2:]

    for video_path in tqdm.tqdm(TRAIN_VIDEO_PATHS):
        video_name = video_path.stem
        image_name_pattern = video_name + "-{:05d}.png"
        with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
            for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
                sink.save_image(image=image)

def check_image_folder():
    image_paths = sv.list_files_with_extensions(
    directory=IMAGE_DIR_PATH,
    extensions=["png", "jpg", "jpg"])

    print('image count:', len(image_paths))

# base_model = GroundedSAM(ontology=ontology)
# dataset = base_model.label(
#     input_folder=IMAGE_DIR_PATH,
#     extension=".png",
#     output_folder=DATASET_DIR_PATH)

