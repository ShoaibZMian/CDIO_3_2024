from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

# ontology = CaptionOntology({
#     "dog": "dog",
#     "backpack": "backpack"
# })

# base_model = GroundedSAM(ontology=ontology)

# # detections = base_model.predict("C:/Users/SkumJustEatMe/CDIO_3_2024/Testing/dog.jpg")

# base_model.label(
#   input_folder="C:/Users/SkumJustEatMe/CDIO_3_2024/Testing",
#   extension=".jpg",
#   output_folder="C:/Users/SkumJustEatMe/CDIO_3_2024/Testing/Label_Data"
# )

target_model = YOLOv8("yolov8n.pt")
target_model.train("C:/Users/SkumJustEatMe/CDIO_3_2024/Testing/Label_Data/data.yaml", epochs=200)