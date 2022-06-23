import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('Data', 'Testiranje')
number_images = 30

captureImage = cv2.VideoCapture(0)
print("Ovde se nalazim")
time.sleep(5)
for image_number in range(number_images):
    #print(captureImage.isOpened())
    print("Sakupljam sliku broj: {}".format(image_number))
    ret, frame = captureImage.read()
    image_file_name = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(image_file_name, frame)
    cv2.imshow('frame', frame)
    time.sleep(0.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captureImage.release()
cv2.destroyAllWindows()

