import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from scipy.spatial.distance import cosine
import tkinter as tk
from tkinter import filedialog
import pygame
import SCINDataset

pygame.init()

driveloc = "D:/"

# Button Class
class Button:
    def __init__(self, surface, imageN, imageP, x, y, scale):
        self.imageN = imageN
        self.imageP = imageP
        self.currentimage = imageN
        self.surfaceW, self.surfaceL = surface.get_size()
        imageW, imageL = imageN.get_size()

        imageW = int(imageW * scale)
        imageL = int(imageL * scale)
        self.imageN = pygame.transform.scale(self.imageN, (imageW, imageL))
        self.imageP = pygame.transform.scale(self.imageP, (imageW, imageL))
        self.currentimage = self.imageN
        self.rect = self.currentimage.get_rect()
        self.rect.topleft = ((self.surfaceW - imageW) // x, (self.surfaceL - imageL) // y)
        self.clicked = False

    def handle_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.clicked = True
                self.currentimage = self.imageP
                return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.clicked = False
            self.currentimage = self.imageN
        return False


# Global State and Window Setup
state = "homepage"
windW, windH = 900, 600
screen = pygame.display.set_mode((windW, windH))
pygame.display.set_caption("Skin Condition Detector")
home = pygame.image.load("Images/Start page.jpg").convert()
home = pygame.transform.scale(home, (windW, windH))
default = pygame.image.load("Images/App navigation page.jpg").convert()
default = pygame.transform.scale(default, (windW, windH))
sourcepage = pygame.image.load("Images/Sources page.jpg").convert()
sourcepage = pygame.transform.scale(sourcepage, (windW, windH))
eczemapage = pygame.image.load("Images/Eczema page.jpg").convert()
eczemapage = pygame.transform.scale(eczemapage, (windW, windH))
notEczemapage = pygame.image.load("Images/Not Eczema page.jpg").convert()
notEczemapage = pygame.transform.scale(notEczemapage, (windW, windH))

# Buttons
startedButton = pygame.image.load("Images/get-started-button.png").convert_alpha()
startedButtonPress = pygame.image.load("Images/get-started-pressed-button.png").convert_alpha()
getStarted = Button(screen, startedButton, startedButtonPress, 8, 4.7, 0.9)

sourceButton = pygame.image.load("Images/sources-button.png").convert_alpha()
sourceButtonPress = pygame.image.load("Images/sources-pressed-button.png").convert_alpha()
sources = Button(screen, sourceButton, sourceButtonPress, 1.17, 4.7, 0.9)

homeButton = pygame.image.load("Images/home-button.png").convert_alpha()
homeButtonPress = pygame.image.load("Images/home-pressed-button.png").convert_alpha()
homeB = Button(screen, homeButton, homeButtonPress, 8, 4.7, 0.9)
homeB2 = Button(screen, homeButton, homeButtonPress, 1.17, 4.7, 0.9)

uploadPhotoButton = pygame.image.load("Images/upload-photo-button.png").convert_alpha()
uploadPhotoButtonPress = pygame.image.load("Images/upload-photo-pressed-button.png").convert_alpha()
uploadPhoto = Button(screen, uploadPhotoButton, uploadPhotoButtonPress, 2, 1.5, 1)

# Tkinter Setup
root = tk.Tk()
root.withdraw()

# Pretrained Model Setup (ResNet)
model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# Preprocessing Function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Feature Extraction
def extract_features(image, model):
    with torch.no_grad():
        features = model(image).squeeze()
    return features.numpy()

# Compare Features
def compare_features(user_features, dataset_features, k=5):
    similarities = [1 - cosine(user_features, features) for features in dataset_features]
    sorted_indices = np.argsort(similarities)[-k:]  # Indices of k-nearest neighbors
    most_similar_scores = [similarities[i] for i in sorted_indices]
    average_similarity = np.mean(most_similar_scores)
    return average_similarity, sorted_indices

# Load SCINDataset
eczema_images = SCINDataset.all_data[SCINDataset.all_data["has_eczema"] == 1]
eczema_features = []

for _, row in eczema_images.iterrows():
    image_path = row["filename"]
    image_path = driveloc + image_path

    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        continue

    try:
        eczema_image = cv2.imread(image_path)
        if eczema_image is None:
            print(f"Error: Unable to read the image file - {image_path}")
            continue

        eczema_image_rgb = cv2.cvtColor(eczema_image, cv2.COLOR_BGR2RGB)
        eczema_tensor = preprocess_image(eczema_image_rgb)
        eczema_features.append(extract_features(eczema_tensor, model))

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Upload Image Functionality
def open_image_file():
    global state  # Ensure the function updates the global 'state' variable
    filepath = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if filepath:
        try:
            userPic = cv2.imread(filepath)
            if userPic is None:
                print("Error: Unable to read the uploaded image.")
                return

            userPic_rgb = cv2.cvtColor(userPic, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(userPic_rgb, (int(userPic.shape[1] * 0.25), int(userPic.shape[0] * 0.25)))
            user_tensor = preprocess_image(resized_image)
            user_features = extract_features(user_tensor, model)

            threshold = 0.23  # Adjust threshold as needed
            k = 5  # Number of neighbors to consider
            average_similarity, _ = compare_features(user_features, eczema_features, k = k)

            if average_similarity > (1 - threshold):  # Convert distance threshold to similarity
                state = "eczema"  # Update state to eczema
            else:
                state = "not eczema"  # Update state to not eczema

        except Exception as e:
            print(f"Error processing the uploaded image: {e}")

# Main Application Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if state == "homepage":
        screen.blit(home, (0, 0))
        for event in pygame.event.get():
            if getStarted.handle_click(event):
                state = "get started"
            if sources.handle_click(event):
                state = "sources"

        screen.blit(getStarted.currentimage, getStarted.rect.topleft)
        screen.blit(sources.currentimage, sources.rect.topleft)
        pygame.display.flip()

    elif state == "get started":
        screen.blit(default, (0, 0))
        for event in pygame.event.get():
            if homeB.handle_click(event):
                state = "homepage"
            if sources.handle_click(event):
                state = "sources"
            if uploadPhoto.handle_click(event):
                open_image_file()  # Update state based on the uploaded image

        screen.blit(homeB.currentimage, homeB.rect.topleft)
        screen.blit(sources.currentimage, sources.rect.topleft)
        screen.blit(uploadPhoto.currentimage, uploadPhoto.rect.topleft)
        pygame.display.flip()

    elif state == "sources":
        screen.blit(sourcepage, (0, 0))
        for event in pygame.event.get():
            if getStarted.handle_click(event):
                state = "get started"
            if homeB2.handle_click(event):
                state = "homepage"

        screen.blit(getStarted.currentimage, getStarted.rect.topleft)
        screen.blit(homeB2.currentimage, homeB2.rect.topleft)
        pygame.display.flip()

    elif state == "eczema":
        screen.blit(eczemapage, (0, 0))
        for event in pygame.event.get():
            if homeB.handle_click(event):
                state = "homepage"
            if sources.handle_click(event):
                state = "sources"

        screen.blit(homeB.currentimage, homeB.rect.topleft)
        screen.blit(sources.currentimage, sources.rect.topleft)
        pygame.display.flip()

    elif state == "not eczema":
        screen.blit(notEczemapage, (0, 0))
        for event in pygame.event.get():
            if homeB.handle_click(event):
                state = "homepage"
            if sources.handle_click(event):
                state = "sources"

        screen.blit(homeB.currentimage, homeB.rect.topleft)
        screen.blit(sources.currentimage, sources.rect.topleft)
        pygame.display.flip()
        

pygame.quit()
