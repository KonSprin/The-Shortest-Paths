from random import randint
import numpy as np
import noise
import cv2
from spmodule.splib import generate_graph, random_points

if False:
  width = 400
  height = 400
  step = 2 
  frame_width = width * step
  frame_height = height * step
  mountain_height = 10
  img = np.zeros((frame_height,frame_width,3), np.float32)

  grass = [34, 128, 55]
  grass2 = [27, 68, 40]
  sand = [175, 214, 238]
  mountain = [137, 137, 139]
  mountain2 = [29, 47, 78]
  snow = [250, 250, 255]

  colours = [grass, grass2, sand, mountain, mountain2, snow]
  colours.reverse()
  levels = [1,0.4,0.3,0.2,0.1,-0.04]
  # levels.reverse()

  scale = width/3
  octaves = 6
  persistence = 0.5
  lacunarity = 2.0
  base = 400
  print(base)
  for i in range(height):
    for j in range(width):
      nois = 0
      for l in [1, 2, 4, 8]:
        nois += noise.pnoise2(i/(scale/l), j/(scale/l), octaves=octaves, 
                             persistence=persistence, lacunarity=lacunarity, 
                             repeatx=1024, repeaty=1024, base=base)/l
      
      # img [i*step:i*step+step,j*step:j*step+step] = 255 - np.uint32((nois + 0.5) * 255)
      img [i*step:i*step+step,j*step:j*step+step] = np.tanh(nois)/2 + 0.5
      # for level, colour in zip(levels,colours):
      #   if nois < level:
      #     img[i*step:i*step+step,j*step:j*step+step] = colour
        
  cv2.imwrite(f"visualizations/test.png", np.uint8(img*100))
  print(img.min())
  print(img.max())
  cv2.imshow('frame', np.uint8(img*scale))

  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
    
    
elif False: 
  width = 200
  height = 200
  step = 4
  frame_width = width * step
  frame_height = height * step
  
  img = np.zeros((frame_height,frame_width,3), np.float32)
  
  images = {}
  for scale in [10, 20, 50, 100]:
    images[scale] = cv2.imread(f"visualizations/noise{scale}reduced.png")
    img += images[scale]

  cv2.imshow('frame', np.uint8(img/2))
  cv2.imwrite(f"visualizations/noise_combined_reduced.png", np.uint8(img/1.875))
  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()


else:
  width = 200
  height = 200
  step = 4
  frame_width = width * step
  frame_height = height * step
  
  img = cv2.imread(f"visualizations/test.png")
  
  grass = [34, 128, 55]
  grass2 = [27, 68, 40]
  sand = [175, 214, 238]
  mountain = [137, 137, 139]
  mountain2 = [29, 47, 78]
  snow = [250, 250, 255]

  colours = [grass, grass2, sand, mountain, mountain2, snow]
  colours.reverse()
  levels = [800,78,70,60,55,45]
  
  print(img[120, 700])
  
  timg = img.copy()
  
  for i in range(height):
    for j in range(width):
      for level, colour in zip(levels,colours):
        if timg[i*step+1,j*step+1,2] < level:
          img[i*step:i*step+step,j*step:j*step+step] = colour
  
  graph = generate_graph(width, height)
  random_points(graph, img, step, 6, 0, 5)
  
  cv2.imshow('mountains', np.uint8(img))
  # cv2.imwrite(f"visualizations/mountains4_walls.png", np.uint8(img))
  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()