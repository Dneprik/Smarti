import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'WoodDataset')
WORK_DIR = os.path.join(BASE_DIR, 'yolo_data')

def prepare_data():

    raw_images_dir = os.path.join(RAW_DATA_DIR, 'images')
    raw_labels_dir = os.path.join(RAW_DATA_DIR, 'labels')

    if not os.path.exists(raw_images_dir):
        print(f"Error, folder {raw_images_dir} doesn't exists ")
        return False

    files = [f for f in os.listdir(raw_images_dir) if f.endswith('.png')]
    if not files:
        print(f"Error, folder {raw_images_dir} doesn't has .png files")

    train_files, val_files = train_test_split(files, test_size=0.2, random_state=29)

    for category in ['train','val']:
        os.makedirs(os.path.join(WORK_DIR, 'images', category), exist_ok=True)
        os.makedirs(os.path.join(WORK_DIR, 'labels', category), exist_ok=True)

    def copy_files(file_list, category):
        for filename in file_list:


            source_image = os.path.join(raw_images_dir, filename)
            destin_image = os.path.join(WORK_DIR, 'images', category, filename)
            shutil.copy(source_image, destin_image)

            onlyname = os.path.splitext(filename)[0]
            source_txt = os.path.join(raw_labels_dir, onlyname + '.txt')
            if os.path.exists(source_txt):
                destin_txt = os.path.join(WORK_DIR, 'labels', category, onlyname + '.txt')
                shutil.copy(source_txt, destin_txt)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    yaml_data = {
        'path': WORK_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'knot'}
    }

    with open(os.path.join(BASE_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f)

    return True

def main():
    print('Preparing data')
    if not prepare_data():
        return

    print('Education model')

    model = YOLO('yolov8n.pt')

    model.train(
        data=os.path.join(BASE_DIR, 'data.yaml'),
        epochs=20,
        imgsz=640,
        project=os.path.join(BASE_DIR, 'runs'),
        name='knot_detector'
    )

    success = model.export(format='onnx', opset=12)

    if success:
        print('Success, file onnx is created')

if __name__ == '__main__':
    main()

