from PIL import ImageFont, ImageDraw, Image
import math
import random
import os

class TrainDataGenerator:
    def __init__(self) -> None:
        self.PLATE_TEMPLATE = None
        self.PLATE_FONT = None
        self.WIDTH_SCALE_COEF = 1
        self.HEIGHT_SCALE_COEF = 1
        self.REAL_PLATE_WIDTH = 520
        self.REAL_PLATE_HEIGHT = 114
        self.REAL_CHAR_HEIGHT = 80
        self.FONT_Y_PADDING = 13
        self.FONT_X_PADDING = 13
        self.BG_size = (50, 50)
        self.load_plate_template('./LICENSE_PLATE_TEMPLATE.jpg')
        self.update_scale_coef()
        self.load_plate_font('./arklatrs.ttf')
    
    def load_plate_template(self, path : str = None) -> None:
        try: 
            with Image.open(path) as image:
                self.PLATE_TEMPLATE = image.copy()
        except FileNotFoundError:
            print("[ERROR]: File not found!")
        finally:
            if self.PLATE_TEMPLATE is not None:
                print("[INFO]:  Template has been loaded succesfully.")
                print("[INFO]:  Template size: ", self.PLATE_TEMPLATE.size)
            else:
                self.PLATE_TEMPLATE = Image.new("RGB", self.BG_size, "white")
                print("[INFO]:  Blank template created.")
                print("[INFO]:  Template size: ", self.PLATE_TEMPLATE.size)

    def load_plate_font(self, path : str = None) -> None:
        font_size = math.floor((self.REAL_CHAR_HEIGHT + 2 * self.FONT_Y_PADDING) * self.HEIGHT_SCALE_COEF)
        if font_size <= 0:
            font_size = 1 + self.PLATE_TEMPLATE.height // 2
        try:
            self.PLATE_FONT = ImageFont.truetype(path, font_size)
        except OSError:
            print("[ERROR]: Font could not be loaded!")
        finally:
            if self.PLATE_FONT is not None:
                print("[INFO]:  Font loaded succesfully.")
                print("[INFO]:  Font size: ", font_size)
            else:
                self.PLATE_FONT = ImageFont.load_default(size = font_size)
                print("[INFO]:  Default font loaded.")
                print("[INFO]:  Font size: ", font_size)

    def update_scale_coef(self):
        template_width, template_height = self.PLATE_TEMPLATE.size
        width_coef = round(template_width / self.REAL_PLATE_WIDTH, 2)
        height_coef = round(template_height / self.REAL_PLATE_HEIGHT, 2)
        self.WIDTH_SCALE_COEF = width_coef
        self.HEIGHT_SCALE_COEF = height_coef

    def create_plate(self, letter : str, template : Image.Image, rng=False):
        draw = ImageDraw.Draw(template)
        x, y = self.BG_size
        x = x // 2 
        y = y // 2 
        font= self.PLATE_FONT
        if rng:
            x+=random.randint(-20, 20)
            y+=random.randint(-15, 15)
            #font=int(font*random.random())
        draw.text((x, y), letter, font = font, fill = 'black', anchor = 'mm')

    def show_image(self, show : bool = False, save : bool = False, name="test.jpg"):
        if show:
            self.PLATE_TEMPLATE.show()
        if save:
            self.PLATE_TEMPLATE.save(name)

    def clear_template(self):
        self.PLATE_TEMPLATE = Image.new("RGB", self.BG_size, "white")

if __name__ == "__main__":
    generator = TrainDataGenerator()

    # Dorobić pętlę do przechodzenia po wszystkich znakach A-Z, 0-9
    ascii_range= list(range(65,91))
    ascii_range+= list(range(48,57))
    for letter_int in ascii_range:
        letter = chr(letter_int)
        if not os.path.exists("data_classificator"):
            os.makedirs("data_classificator")
        if not os.path.exists("data_classificator/"+letter):
            os.makedirs("data_classificator/"+letter)
        # Do create_plate() można dodać augmentację
        # i ewentualnie dopisać losowanie nowej wielkości liter
        # np. przez mnożenie self.PLATE_FONT przez jakiś współczynnik
        # z przedziału [0, 1] (tylko nie nadpisywać self.PLATE_FONT)
        for i in range(20):
            if i==0:
                generator.create_plate(letter, generator.PLATE_TEMPLATE,rng=False)
            else:
                generator.create_plate(letter, generator.PLATE_TEMPLATE,rng=True)
            path="data_classificator/"+letter+f"/{i}.jpg"
            generator.show_image(False, True,path)
            generator.clear_template()