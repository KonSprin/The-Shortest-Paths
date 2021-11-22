import numpy as np
import noise
import cv2

width = 300
height = 200
step = 4
frame_width = width * step
frame_height = height * step
mountain_height = 10
img = np.zeros((frame_height,frame_width,3), np.uint32)

grass = [34, 128, 55]
grass2 = [27, 68, 40]
sand = [175, 214, 238]
mountain = [137, 137, 139]
mountain2 = [29, 47, 78]
snow = [250, 250, 255]

colours = [grass, grass2, sand, mountain, mountain2, snow]
colours.reverse()
steps = [1,0.4,0.3,0.2,0.1,-0.04]
# steps.reverse()

scale = 50
octaves = 6
persistence = 0.5
lacunarity = 2.0
for i in range(height):
  for j in range(width):
    nois = noise.pnoise2(i/scale, j/scale, octaves=octaves, 
                                  persistence=persistence, lacunarity=lacunarity, 
                                  repeatx=1024, repeaty=1024, base=0)
    
    for stepe, colour in zip(steps,colours):
      if nois < stepe:
        img[i*step:i*step+step,j*step:j*step+step] = colour
      
cv2.imwrite("visualizations/mountains.png", np.uint8(img))

cv2.imshow('frame', np.uint8(img))

if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()