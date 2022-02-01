import eval
import train

inp = int(input('1 - train mode\n2 - eval mode\n'))

if inp == 1:
    train.train()
elif inp == 2:
    eval.eval()
else:
    print('wrong input')
