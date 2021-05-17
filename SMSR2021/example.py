#import cv2 as cv
#import os 

# img_file = "testsets/benchmark/videos_img/LR_bicubic"

# imgs = os.listdir(img_file)
# for i in range(len(imgs)):
#     img_0 = cv.imread(img_file + "/" + imgs[i])
#     H,W = img_0.shape[0],img_0.shape[1]
#     img_0 = cv.resize(img_0,(W * 2, H * 2))
#     cv.imwrite("testsets/benchmark/videos_img/HR/" + imgs[i],img_0)
#     cv.waitKey(0)


import cv2
import os
fps = 24
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(filename='\experiment\SMSR_X2\\results\\result_video\\result.mp4', fourcc=fourcc, fps=fps, frameSize=(1280, 960))  # 图片实际尺寸，不然生成的视频会打不开
for i in range(1,6764):
  p = i
  
  if os.path.exists('D:\Orange\Code\other\github\SMSR2021\experiment\SMSR_X2\\results'):  #判断图片是否存在
    print(p)
    img = cv2.imread(filename='D:\Orange\Code\other\github\SMSR2021\experiment\SMSR_X2\\results\\pic ('+str(p)+').png')
    cv2.waitKey(1)
    video_writer.write(img)
video_writer.release()
