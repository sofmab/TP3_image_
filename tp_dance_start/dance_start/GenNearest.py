
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
         """ Generate image from skeleton """
         closest_distance = float('inf')
         closest_img = None

         for idx in range(self.videoSkeletonTarget.skeCount()):
             skeleton = self.videoSkeletonTarget.ske[idx]
            
             distance = ske.distance(skeleton)
            
            # prendre la distance minimale et récupérer l'image correspendante
             if distance < closest_distance:
                 closest_distance = distance
                 closest_img = self.videoSkeletonTarget.readImage(idx)
        
         return closest_img     
         '''
         empty = np.ones((64,64, 3), dtype=np.uint8)
         return empty'''




