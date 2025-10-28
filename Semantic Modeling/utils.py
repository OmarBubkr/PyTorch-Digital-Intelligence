from collections import namedtuple


cs_labels = namedtuple('CityscapesClass', ['name', 'train_id', 'color'])
cs_classes = [
    cs_labels('road',          0, (128, 64, 128)),
    cs_labels('sidewalk',      1, (244, 35, 232)),
    cs_labels('building',      2, (70, 70, 70)),
    cs_labels('wall',          3, (102, 102, 156)),
    cs_labels('fence',         4, (190, 153, 153)),
    cs_labels('pole',          5, (153, 153, 153)),
    cs_labels('traffic light', 6, (250, 170, 30)),
    cs_labels('traffic sign',  7, (220, 220, 0)),
    cs_labels('vegetation',    8, (107, 142, 35)),
    cs_labels('terrain',       9, (152, 251, 152)),
    cs_labels('sky',          10, (70, 130, 180)),
    cs_labels('person',       11, (220, 20, 60)),
    cs_labels('rider',        12, (255, 0, 0)),
    cs_labels('car',          13, (0, 0, 142)),
    cs_labels('truck',        14, (0, 0, 70)),
    cs_labels('bus',          15, (0, 60, 100)),
    cs_labels('train',        16, (0, 80, 100)),
    cs_labels('motorcycle',   17, (0, 0, 230)),
    cs_labels('bicycle',      18, (119, 11, 32)),
    cs_labels('ignore_class', 19, (0, 0, 0)),
]

train_id_to_color = [c.color for c in cs_classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)
