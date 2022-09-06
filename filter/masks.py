from pathlib import Path
from PIL import Image

def create_mask(img, mask, value=255):
    mask = Image.new('L', img.size, color=0)
    for x in range(20, img.width-30):
        for y in range(10, img.height-17):
            mask.putpixel((x,y), value)
    return mask

def addmask_to_picture(img, maskval=47):
    for y in range(img.height):
        for x in range(20):
            img.putpixel((x,y), (maskval, maskval, maskval)) 
        for x in range(img.width-20,img.width):
            img.putpixel((x,y), (maskval, maskval, maskval)) 
    for x in range(img.width):
        for y in range(20):
            img.putpixel((x,y), (maskval, maskval, maskval)) 
        for y in range(img.height-20,img.height):
            img.putpixel((x,y), (maskval, maskval, maskval)) 
    return img
    
if __name__=='__main__':
    #process_testimage()
    testset_folder = Path(__file__).parent.parent / 'testdata/test/render0/image0.png'
    img = Image.open(testset_folder)
    ret = addmask_to_picture(img)
    ret.save("out.png")
    