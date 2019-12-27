# Data Gathering and Processing

To train a custom object detection model from scratch on your own dataset, follow these steps on your local computer

1. Gather all the images and put a split the images into the `data/train` and `data/test` directories. Around 50 images per class is enough or more than 50 if you have only 1 class or so.
2. Run labelImg.exe application and start annotating the images. Then edit `data/predefined_classes.txt` file with your class names all separated by a new line. In labelImg, click on Open Dir and navigate to the train and test folders. Draw bounding boxes and label them and save them to the same directories as the images. Note: You will have to save for each image. To speed up workflow, use `w` key to draw a box, `d` key to move on to next image and `ctrl + s` to save. If you can't figure out how to use labelImg, you can just google it and you will find tons of resources.
3. Once you are done with annotating, open the [Preprocess_Data.ipynb](Preprocess_Data.ipynb) notebook for further instructions.
