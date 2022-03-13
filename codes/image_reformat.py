# resizing images to be 3x3 degree (84x84) with padding 

from PIL import Image
import os 

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

path = "/Users/ngayulo/Documents/original"

for root, dirs, files in os.walk(path):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(os.path.join(root,dir)):
                for subdir in subdirs:
                    for subsubroot, subsubdirs, subsubfiles in os.walk(os.path.join(subroot,subdir)):
                        for file in subsubfiles:
                            if file.endswith('.png'):
                                print(os.path.join(subsubroot,file))
                                image = Image.open(os.path.join(subsubroot,file))
                                new_size = (84,84)
                                image = image.resize(new_size)
                                new_image = add_margin(image,70,70,70,70,(255,255,255))

                                saving_path = "/Users/ngayulo/Documents/resized_white_bg/"+dir+"/"+subdir
                                if not os.path.exists(saving_path):
                                    os.makedirs(saving_path)
                                new_image.save(saving_path+"/"+file, quality=95)



