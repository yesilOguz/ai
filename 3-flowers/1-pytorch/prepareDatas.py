# 0 - daisy - [1, 0, 0, 0, 0]
# 1 - dandelion - [0, 1, 0, 0, 0]
# 2 - rose - [0, 0, 1, 0, 0]
# 3 - sunflower - [0, 0, 0, 1, 0]
# 4 - tulip - [0, 0, 0, 0, 1]

from PIL import Image
import numpy as np
import os

from sklearn.model_selection import train_test_split

# paths
main = 'flowers/'

daisyPath = main + 'daisy/'
dandelionPath = main + 'dandelion/'
rosePath = main + 'rose/'
sunflowerPath = main + 'sunflower/'
tulipPath = main + 'tulip/'

# filenames in the path
daisyNames = [f for f in os.listdir(daisyPath) if os.path.isfile(os.path.join(daisyPath, f))]
dandelionNames = [f for f in os.listdir(dandelionPath) if os.path.isfile(os.path.join(dandelionPath, f))]
roseNames = [f for f in os.listdir(rosePath) if os.path.isfile(os.path.join(rosePath, f))]
sunflowerNames = [f for f in os.listdir(sunflowerPath) if os.path.isfile(os.path.join(sunflowerPath, f))]
tulipNames = [f for f in os.listdir(tulipPath) if os.path.isfile(os.path.join(tulipPath, f))]

def getArrays(paths, names):
    # empty arrays
    ImgArr = []
    labelArr = []

    daisy = [1, 0, 0, 0, 0]
    dandelion = [0, 1, 0, 0, 0]
    rose = [0, 0, 1, 0, 0]
    sunflower = [0, 0, 0, 1, 0]
    tulip = [0, 0, 0, 0, 1]

    size = (64, 64)

    for i in range(len(paths)):
        for name in names[i]:
            img = np.asarray(Image.open(paths[i]+name).resize(size))
            ImgArr.append((img.reshape(3, img.shape[0], img.shape[1]))/255)

            flower = paths[i].split('/')[1]
            
            if flower == 'daisy':
                labelArr.append(np.asarray(daisy))
            elif flower == 'dandelion':
                labelArr.append(np.asarray(dandelion))
            elif flower == 'rose':
                labelArr.append(np.asarray(rose))
            elif flower == 'sunflower':
                labelArr.append(np.asarray(sunflower))
            elif flower == 'tulip':
                labelArr.append(np.asarray(tulip))

    print(np.asarray(ImgArr[50]).shape)
    
    #      Images            Labels
    return np.array(ImgArr), np.array(labelArr)

x, y = getArrays([daisyPath, dandelionPath, rosePath, sunflowerPath, tulipPath],
                 [daisyNames, dandelionNames, roseNames, sunflowerNames, tulipNames])

x, testX, y, testY = train_test_split(x, y, test_size=0.33, random_state=42)

# save arrays as numpy file
np.save('trainX.npy', x) # Images
np.save('trainY.npy', y) # Labels

np.save('testX.npy', testX) # Test Images
np.save('testY.npy', testY) # Test Labels

            
