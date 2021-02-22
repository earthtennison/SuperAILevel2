import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread(r"D:\SuperAILevel2\week5\xxx.png")
print(im.shape)
# imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# print(imGray.shape)



# imCut = im[0:512,0:100] #heigth * width
# imcutG = cv2.cvtColor(imCut,cv2.COLOR_BGR2GRAY)

def skyDetect(image):

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = np.ones((512,512,1))
    h2=[]
    v2=[]
    for x in range(0,512,1):
        col_pix = []
        col_pix.append(imageGray[0][x])
        is_skyline = False
        for y in range(1,512):

            pixel = imageGray[y][x] #from top to bottom
            # 255 = white, 0 = black
            col_pix.append(pixel)
            # print(pixel)

            # grey jump
            if np.abs(col_pix[y] - col_pix[y-1] )> 30 and col_pix[y] <100:
                # jump buffer
                for count in range(1,4): # 1-5
                        if col_pix[y] > 100: #found white
                            is_skyline = False
                            break
                        y += 1
                        pixel = imageGray[y][x]
                        col_pix.append(pixel)
                        is_skyline = True
                        print("y",y)
                        print("count",count)


                if is_skyline :
                    # image[y][x] = (0, 255, 0)
                    cv2.circle(image, (x, y-5), 5, (0, 255, 0), 1)
                    result[:y,x] = 0
                    h2.append(x)
                    v2.append(y)
                    break

        # print(col_pix)

    return image,result,h2,v2

# cv2.imshow("imgra", imGray)
# cv2.imshow("im",im)
detect,res,point,_ = skyDetect(im)
print(point)
cv2.imshow("detect",detect)
cv2.imshow("res2",res)
cv2.waitKey(0)

