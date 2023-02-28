import json

from visualisation import Recognizer
import cv2
import os

# to use this module you should create folder train and inside create folder for each person.
# The tree should look like this
#SRC\TRAIN
#├───vanya
#│       photo_2023-02-28_15-40-31 (2).jpg
#│       photo_2023-02-28_15-40-31 (3).jpg
#│       photo_2023-02-28_15-40-31.jpg
#│
#└───vova
#        2YNyHzMBKZWYKVpU1HhpObqLuZFFRnlsX5V5ubHIbt9_5B3W4dAhPcNcaO5ISKpjvEjpR6g9D9vfwEZ89oS86L2v.jpg
#        i8VdDppqITWBM3Rd67Vw-B2V1PViwKso4mcCDtQtmvUIzXcShoZL9-5TzHjDb9TWlZ4qRjCYVtR7FmEyV9kU3gvu.jpg
#        lOm3kgFIf9d0vZk6PpIej3rnQjr_OlB-Fhzv-eLj7I_mHp96DxnBK-KVwCG1v0awAQyVJ-ejNcuFIm5Lptdts0Z-.jpg

if __name__ == "__main__":
    index_to_name = dict()
    name_to_index = dict()

    recognizer = Recognizer()

    for root, dirs, files in os.walk("train"):
        for file in files:
            _, name = root.split(os.sep)

            index = len(index_to_name)

            if name in name_to_index:
                index = name_to_index[name]
            else:
                index_to_name[index] = name
                name_to_index[name] = index

            #load image
            image = cv2.imread(root + "\\" + file)
            result = recognizer.process(image)
            if len(result) == 1:
                h = result[0]
                recognizer.database.append([h.features, index, name])
                print("SUCCESS:", root, file)
            else:
                print("ERROR  :", root, file)

    with open("database.json", "w") as f:
        f.write(json.dumps(recognizer.database))
