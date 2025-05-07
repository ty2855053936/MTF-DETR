import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/MTF-DETR.yaml')
    # model.load('') # loading pretrain weights')
    # model.load('') # loading pretrain weights
    model.train(data='dataset/data.yaml',
                cache=False,
                epochs=250,
                batch=4,
                workers=4,
                #resume='last.pt path', # last.pt path
                project='runs/train',
                name='MTF-DETR',
                )