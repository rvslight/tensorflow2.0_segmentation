import os
import sys
import numpy as np

def prepare_train_directories(config):
    out_dir = config.TRAIN_DIR+config.RECIPE
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def mask2contour(mask, width=1):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)


def annotate_to_images(images, labels, preds,thresh=0.5):
  preds = preds>=thresh
  # preds[preds > thresh] = 1
  # preds[preds <= thresh] = 0
  annotated_images = []
  for item in zip(images, labels, preds):
    image = item[0]
    mask = item[1]
    pred = item[2]
    #
    ### image ###
    if image.shape[0] == 3:
      image = np.transpose(image, [1, 2, 0])

    image = input_to_image(image)

    # mask = np.transpose(mask, [1, 2, 0])
    # pred = np.transpose(pred, [1, 2, 0])

    image = image.astype('uint8')
    for index in range(1):
        image_with_mask = image.copy()
        mask_line, pred_line = mask2contour(mask[:,:,index]), mask2contour(pred[:,:,index])

        image_with_mask[mask_line == 1, :2] = 0
        image_with_mask[pred_line == 1, :1] = 255

        ## one channel
        # image_with_mask[np.expand_dims(mask_line,axis=0) == 1] = 0
        # image_with_mask[np.expand_dims(pred_line,axis=0) == 1] = 255
        del mask_line, pred_line
        image_with_mask = np.transpose(image_with_mask, [2, 0, 1]) # change to C,H,W

        annotated_images.append(image_with_mask)

        ## one_channel
        # annotated_images.append(image_with_mask) # C,H,W order..
    del mask, pred
  return annotated_images

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
def input_to_image(input):#, rgb_mean=MEAN, rgb_std=STD):
    #input h,w,c
    input = (input+1) / 2
    image = (input - np.min(input)) / (np.max(input) - np.min(input))*255


    # input = input * 0.5 + 0.5
    # # input[:,:,0] = (input[:,:,0]*rgb_std[0]+rgb_mean[0])
    # # input[:,:,1] = (input[:,:,1]*rgb_std[1]+rgb_mean[1])
    # # input[:,:,2] = (input[:,:,2]*rgb_std[2]+rgb_mean[2])
    # image = (input*255).astype(np.uint8)
    return image