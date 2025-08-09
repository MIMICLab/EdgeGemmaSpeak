#!/usr/bin/env python3
"""
Test script for ARIA display functionality without actual ARIA connection
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import random
import time

class TestAriaDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ARIA Input Display Test")
        
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Original view label
        tk.Label(main_frame, text="Original View", font=("Arial", 12, "bold")).grid(row=0, column=0, padx=5)
        self.original_label = tk.Label(main_frame)
        self.original_label.grid(row=1, column=0, padx=5)
        
        # Cropped view label
        tk.Label(main_frame, text="Cropped View (896x896)", font=("Arial", 12, "bold")).grid(row=0, column=1, padx=5)
        self.cropped_label = tk.Label(main_frame)
        self.cropped_label.grid(row=1, column=1, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Testing display...", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # Simulated gaze position
        self.gaze_x = 640
        self.gaze_y = 360
        self.gaze_vx = random.randint(-5, 5)
        self.gaze_vy = random.randint(-5, 5)
        
        # Start updating
        self.update_display()
        
    def create_test_image(self):
        """Create a test image with pattern"""
        # Create a gradient pattern
        width, height = 1280, 720
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Draw gradient background
        for i in range(width):
            color = int(255 * i / width)
            draw.line([(i, 0), (i, height)], fill=(color, 100, 255-color))
        
        # Draw grid
        for x in range(0, width, 100):
            draw.line([(x, 0), (x, height)], fill=(0, 0, 0), width=1)
        for y in range(0, height, 100):
            draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=1)
            
        # Add some text
        draw.text((width//2 - 50, height//2 - 10), "ARIA TEST", fill=(255, 255, 255))
        
        return img
    
    def add_gaze_point(self, image):
        """Add green gaze point to image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Draw green circle at gaze point
        draw.ellipse([(self.gaze_x-12, self.gaze_y-12), 
                      (self.gaze_x+12, self.gaze_y+12)], 
                     outline=(0, 255, 0), width=3)
        draw.ellipse([(self.gaze_x-4, self.gaze_y-4), 
                      (self.gaze_x+4, self.gaze_y+4)], 
                     fill=(0, 255, 0))
        
        # Draw crosshair
        draw.line([(self.gaze_x-20, self.gaze_y), (self.gaze_x+20, self.gaze_y)], 
                  fill=(0, 255, 0), width=2)
        draw.line([(self.gaze_x, self.gaze_y-20), (self.gaze_x, self.gaze_y+20)], 
                  fill=(0, 255, 0), width=2)
        
        return img_copy
    
    def crop_at_gaze(self, image, crop_size=896):
        """Crop image around gaze point"""
        width, height = image.size
        half_size = crop_size // 2
        
        # Calculate crop boundaries
        left = max(0, self.gaze_x - half_size)
        top = max(0, self.gaze_y - half_size)
        right = min(width, self.gaze_x + half_size)
        bottom = min(height, self.gaze_y + half_size)
        
        # Adjust if crop would go out of bounds
        if right - left < crop_size:
            if left == 0:
                right = min(width, crop_size)
            else:
                left = max(0, width - crop_size)
                
        if bottom - top < crop_size:
            if top == 0:
                bottom = min(height, crop_size)
            else:
                top = max(0, height - crop_size)
        
        # Crop the image
        cropped = image.crop((left, top, right, bottom))
        
        # Resize to exact size if needed
        if cropped.size != (crop_size, crop_size):
            cropped = cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
            
        return cropped
    
    def update_gaze_position(self):
        """Update simulated gaze position"""
        # Move gaze point
        self.gaze_x += self.gaze_vx
        self.gaze_y += self.gaze_vy
        
        # Bounce off edges
        if self.gaze_x <= 50 or self.gaze_x >= 1230:
            self.gaze_vx = -self.gaze_vx
        if self.gaze_y <= 50 or self.gaze_y >= 670:
            self.gaze_vy = -self.gaze_vy
            
        # Add some randomness
        if random.random() < 0.1:
            self.gaze_vx = random.randint(-5, 5)
            self.gaze_vy = random.randint(-5, 5)
    
    def update_display(self):
        """Update the display with test images"""
        # Update gaze position
        self.update_gaze_position()
        
        # Create test image
        test_img = self.create_test_image()
        
        # Add gaze point
        img_with_gaze = self.add_gaze_point(test_img)
        
        # Display original with gaze
        display_original = img_with_gaze.copy()
        display_original.thumbnail((640, 480), Image.Resampling.LANCZOS)
        photo_original = ImageTk.PhotoImage(display_original)
        self.original_label.configure(image=photo_original)
        self.original_label.image = photo_original
        
        # Crop and display
        cropped_img = self.crop_at_gaze(test_img)
        display_cropped = cropped_img.copy()
        display_cropped.thumbnail((448, 448), Image.Resampling.LANCZOS)
        photo_cropped = ImageTk.PhotoImage(display_cropped)
        self.cropped_label.configure(image=photo_cropped)
        self.cropped_label.image = photo_cropped
        
        # Update status
        self.status_label.configure(text=f"Gaze position: ({self.gaze_x}, {self.gaze_y})")
        
        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("Testing ARIA display functionality...")
    print("This shows how the GUI would look with ARIA input")
    print("The green dot represents the gaze point")
    print("The right panel shows the 896x896 cropped region")
    
    test = TestAriaDisplay()
    test.run()