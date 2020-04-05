import os

def generate_list(images_path, annotations_path, destdir, list_type):

    images = [ f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    data_list = open(destdir + "/" + list_type +'.txt', 'w') # IMG list

    for i, image in enumerate(images):
        annotations = image.split('.')[0] + '.png'
        line = images_path + image + " " + annotations_path + annotations
        data_list.write(line + '\n')

    data_list.close()

imgs_path = "/isilon/Datasets/HumanParsing/LV-MHP-v2/train/images/"
annos_path = "/isilon/Datasets/HumanParsing/LV-MHP-v2/train/clothes-annos/"
dest = "/Users/shira/research-clothes-segmentation/HRNet-Semantic-Segmentation/data/list/mhp"

generate_list(imgs_path, annos_path, dest, "train")
