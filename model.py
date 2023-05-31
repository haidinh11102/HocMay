import cv2
from PIL import Image
from pytesseract import pytesseract

camera = cv2.VideoCapture(1)

while True:
    _,imageFrame = camera.read()
    cv2.imshow('Text detection',imageFrame)
    if cv2.waitKey(1) & 0xff == ord('s'):
        cv2.imwrite('test.jpg',imageFrame)
        break
camera.release()
cv2.destroyAllWindows()

def tesseract():
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"	
    Imagepath = 'test.jpg'
    pytesseract.tesseract_cmd = path_to_tesseract
    test = pytesseract.image_to_string(Image.open(Imagepath))
    print(test[: -1])
tesseract()
