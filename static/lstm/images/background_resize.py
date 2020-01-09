import cv2
import os

AUTHORS = ["Albeniz", "Beethoven", "Chopin", "Grieg", "Mendelssohn", "Tschaikowsky"]

for file in os.listdir():
    if file[:file.find(".")] in AUTHORS and file[:file.find(".")] + "_thumbnail.jpg" not in [file for file in os.listdir() if "background" in file]:
        img = cv2.imread(file)
        cv2.imwrite(file[:file.find(".")] + "_background.jpg", cv2.resize(img, (1000, 1300), interpolation=cv2.INTER_AREA))

