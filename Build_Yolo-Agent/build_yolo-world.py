from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
from ultralytics import YOLOWorld
import multiprocessing

def train():
    data = dict(
        train=dict(
            grounding_data=[
                dict(
                    img_path="./data/image",
                    json_file="./data/final_flickr_separateGT_train.json",
                ),
            ],
        ),
        val=dict(yolo_data=["lvis.yaml"]),
    )
    model = YOLOWorld("yolov8n-worldv2.yaml")
    model.train(data=data, batch=16, epochs=5, trainer=WorldTrainerFromScratch)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
