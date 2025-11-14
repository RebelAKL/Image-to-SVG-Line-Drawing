"""
Generate synthetic test images for the test_images dataset.
This script creates simple PNG images for each category/subcategory using Pillow.
Run: python scripts/generate_test_images.py
"""
from PIL import Image, ImageDraw
import os
import math

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(BASE_DIR, 'test_images')

os.makedirs(TEST_DIR, exist_ok=True)

# Helper builders

def save(img, rel_path):
    out_path = os.path.join(TEST_DIR, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    print('WROTE', out_path)


def make_blank(w, h, color=(255,255,255)):
    return Image.new('RGB', (w,h), color)

# Mechanical parts
# gear: circle with teeth
w,h = 800,600
img = make_blank(w,h,(240,240,240))
d = ImageDraw.Draw(img)
cx,cy = w//2, h//2
r = 150
# draw teeth
teeth = 20
for i in range(teeth):
    ang = 2*math.pi*i/teeth
    x1 = cx + int((r-5)*math.cos(ang))
    y1 = cy + int((r-5)*math.sin(ang))
    x2 = cx + int((r+25)*math.cos(ang+math.pi/teeth))
    y2 = cy + int((r+25)*math.sin(ang+math.pi/teeth))
    d.polygon([(x1,y1),(x2,y2)], fill=(200,200,200))
# inner circle and bore
d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(0,0,0), width=3)
d.ellipse([cx-30, cy-30, cx+30, cy+30], outline=(0,0,0), width=3)
save(img, 'mechanical_parts/gear_spur_01.png')

# bolt: hexagon head + shaft
img = make_blank(600,400,(255,255,255))
d = ImageDraw.Draw(img)
head_c = (200,200,200)
hx,hy = 150,200
hh = 60
hexagon = []
for i in range(6):
    ang = 2*math.pi*i/6
    hexagon.append((hx+int(hh*math.cos(ang)), hy+int(hh*math.sin(ang))))
d.polygon(hexagon, fill=head_c, outline=(0,0,0))
d.rectangle([220,170,500,230], fill=(180,180,180), outline=(0,0,0))
save(img, 'mechanical_parts/bolt_hex_01.png')

# screw: shaft with thread-like lines
img = make_blank(500,300,(255,255,255))
d = ImageDraw.Draw(img)
d.rectangle([200,50,260,250], fill=(220,220,220), outline=(0,0,0))
for y in range(55,245,6):
    d.line([(202,y),(258,y+4)], fill=(100,100,100), width=2)
save(img, 'mechanical_parts/screw_01.png')

# Architectural elements
img = make_blank(1000,600,(245,245,245))
d = ImageDraw.Draw(img)
# grid of windows
for row in range(4):
    for col in range(8):
        x = 50 + col*110
        y = 50 + row*120
        d.rectangle([x,y,x+80,y+90], outline=(0,0,0), fill=(255,255,255))
save(img, 'architectural_elements/facade_grid_01.png')

# floorplan like simple walls
img = make_blank(800,600,(255,255,255))
d = ImageDraw.Draw(img)
d.line([(50,50),(750,50),(750,550),(50,550),(50,50)], fill=(0,0,0), width=6)
d.line([(200,50),(200,400)], fill=(0,0,0), width=6)
d.rectangle([220,420,500,540], outline=(0,0,0), width=4)
save(img, 'architectural_elements/floorplan_01.png')

# Circuit boards
img = make_blank(800,600,(20,60,20))
d = ImageDraw.Draw(img)
# traces
for i in range(10):
    y = 50 + i*40
    d.line([(50,y),(750,y)], fill=(200,200,50), width=6)
# pads
for x in range(120,700,120):
    d.rectangle([x,260,x+20,300], fill=(220,220,220))
save(img, 'circuit_boards/pcb_traces_01.png')

# IC
img = make_blank(600,400,(30,30,30))
d = ImageDraw.Draw(img)
d.rectangle([200,100,400,300], fill=(10,10,10), outline=(200,200,200))
for i in range(8):
    d.rectangle([190,110+i*20,200,120+i*20], fill=(180,180,180))
    d.rectangle([400,110+i*20,410,120+i*20], fill=(180,180,180))
save(img, 'circuit_boards/ic_chip_01.png')

# Tools
img = make_blank(800,600,(255,255,255))
d = ImageDraw.Draw(img)
# wrench: handle + crescent
d.rectangle([100,250,500,320], fill=(200,200,200), outline=(0,0,0))
d.pieslice([480,200,680,360], start=300, end=60, fill=(200,200,200), outline=(0,0,0))
save(img, 'tools/wrench_01.png')

# screwdriver
img = make_blank(600,400,(255,255,255))
d = ImageDraw.Draw(img)
d.rectangle([260,80,300,320], fill=(150,150,150), outline=(0,0,0))
d.rectangle([200,40,360,80], fill=(80,40,10))
save(img, 'tools/screwdriver_01.png')

# Edge cases
# high contrast
img = make_blank(800,600,(255,255,255))
d = ImageDraw.Draw(img)
d.ellipse([200,150,600,500], fill=(0,0,0))
save(img, 'edge_cases/high_contrast/black_on_white_01.png')

# low contrast
img = make_blank(800,600,(230,230,230))
d = ImageDraw.Draw(img)
d.rectangle([200,150,600,450], fill=(210,210,210))
save(img, 'edge_cases/low_contrast/soft_box_01.png')

# complex geometry: lots of small overlapping shapes
img = make_blank(1200,800,(255,255,255))
d = ImageDraw.Draw(img)
import random
for i in range(300):
    r = random.randint(5,30)
    x = random.randint(0,1200)
    y = random.randint(0,800)
    d.ellipse([x-r,y-r,x+r,y+r], outline=(0,0,0), width=1)
save(img, 'edge_cases/complex_geometry/complex_01.png')

# simple geometry
img = make_blank(400,300,(255,255,255))
d = ImageDraw.Draw(img)
d.rectangle([50,50,150,150], outline=(0,0,0), width=3)
d.ellipse([200,50,300,150], outline=(0,0,0), width=3)
save(img, 'edge_cases/simple_geometry/simple_shapes_01.png')

# high resolution
img = make_blank(4000,3000,(240,240,240))
d = ImageDraw.Draw(img)
for i in range(50):
    d.line([(0,i*60),(4000,i*60)], fill=(200,200,200), width=3)
save(img, 'edge_cases/high_resolution/high_res_01.png')

# low resolution
img = make_blank(320,240,(255,255,255))
d = ImageDraw.Draw(img)
d.rectangle([40,40,280,200], outline=(0,0,0))
save(img, 'edge_cases/low_resolution/low_res_01.png')

# noisy background
img = make_blank(800,600,(200,200,200))
d = ImageDraw.Draw(img)
for i in range(2000):
    x = random.randint(0,799)
    y = random.randint(0,599)
    col = random.randint(0,255)
    d.point((x,y), fill=(col,col,col))
d.rectangle([200,150,600,450], outline=(0,0,0), width=3)
save(img, 'edge_cases/noisy_background/noisy_01.png')

# minimal features
img = make_blank(400,300,(255,255,255))
d = ImageDraw.Draw(img)
d.line([(50,150),(350,150)], fill=(0,0,0), width=2)
save(img, 'edge_cases/minimal_features/single_line_01.png')

print('Done generating sample images')
