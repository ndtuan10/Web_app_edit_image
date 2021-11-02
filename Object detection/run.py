from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

img_names = ['image/dog.jpg', 'image/office.jpg', 'image/my.jpg', 'image/ugreen_3.jpg',]
for img in img_names: display(Image.open(img))