import torch
from torch.autograd import Variable

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import warnings

def eval():
    warnings.filterwarnings('ignore')
    
    try:
        model = torch.load('best.model')
    except:
        print('i can\'t find your model')
        return
    
    # predict mode
    model.eval()

    while True:    
        #take path of image for use in the predict
        imPath = input('path of your image: ')

        size = (64, 64)
        
        try:
            #open image
            image = Image.open(imPath)
        except:
            print("there is a problem in the path of the photo")    
            continue
                
        #resize and grayscale image
        im = np.array(image.resize(size))
        
        im = torch.Tensor((im.reshape(1, 3, im.shape[0], im.shape[1]))/255)
        im = Variable(im)

        predict = model(im.float())
        
        print(predict.data)
        _, pred = torch.max(predict.data, 1)


        text = ""
        for i in range(len(pred)):
            if pred[i].item() == 0:
                flower = 'daisy'
            elif pred[i].item() == 1:
                flower = 'dandelion'
            elif pred[i].item() == 2:
                flower = 'rose'
            elif pred[i].item() == 3:
                flower = 'sunflower'
            elif pred[i].item() == 4:
                flower = 'tulip'

            text += '{}% - {}\n'.format(int(_[i].item()*100), flower)
            
        #plot the predict and image
        plt.imshow(image)
        plt.title(text, fontsize=15)
        plt.axis("off")

        plt.show()

