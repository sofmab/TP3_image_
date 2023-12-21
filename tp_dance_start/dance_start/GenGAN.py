
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1), 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0), 
            nn.Sigmoid()

        )


    def forward(self, input):

        return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=10):
        
        optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.002, betas=(0.5, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(n_epochs):
            for i, (skeleton, real_image) in enumerate(self.dataloader):
                #entraînement du Discriminateur
                optimizer_D.zero_grad()

                #générer des fausses images avec le générateur
                generated_image = self.netG(skeleton)

                #calculer la perte du discriminateur
                validity_real = self.netD(real_image)
                validity_fake = self.netD(generated_image.detach())
                real_labels = torch.full((real_image.size(0),), self.real_label, dtype=torch.float, device="cpu")
                fake_labels = torch.full((real_image.size(0),), self.fake_label, dtype=torch.float, device="cpu")
                d_loss_real = criterion(validity_real.squeeze(), real_labels)
                d_loss_fake = criterion(validity_fake.squeeze(), fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

                d_loss.backward()
                optimizer_D.step()

                #entraînement du Générateur
                optimizer_G.zero_grad()
                validity = self.netD(generated_image)

                # calculer la perte du générateur
                g_loss = criterion(validity.squeeze(), real_labels)
                g_loss.backward()
                optimizer_G.step()

                # affficher les pertes pour le suivi de l'entraînement
                if i % 100 == 0:
                    print(
                        f"[Epoch ({epoch +1}/{n_epochs})] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                    )
        




    def generate(self, ske):          
        """ generator of image from skeleton """
        pass
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "tp/dance/data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        filename = "data/taichi1.mp4"
        gen = GenGAN(targetVideoSke, False)
        gen.train(200)
        torch.save(gen.netG, gen.filename)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

