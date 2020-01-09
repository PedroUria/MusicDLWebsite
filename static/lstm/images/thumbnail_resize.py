import cv2
import os

AUTHORS = ["Albeniz", "Beethoven", "Chopin", "Grieg", "Mendelssohn", "Tschaikowsky"]

for file in os.listdir():
    if file[:file.find(".")] in AUTHORS and file[:file.find(".")] + "_thumbnail.jpg" not in [file for file in os.listdir() if "thumbnail" in file]:
        img = cv2.imread(file)
        cv2.imwrite(file[:file.find(".")] + "_thumbnail.jpg", cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA))

