## Try #1 - Training the Fer2013 Dataset

#### Observations

What makes this different from other image datasets like the *CIFAR-10* is the **lack of differentiation between features of a certain class**. When sorting images into basic classes like *airplanes* or *cars*, there is a more *noticeable* difference then **slight facial clues that tell a human another individual's emotion**.

Many people would say that this problem can be solved relatively easily. However *this is not the case* because of the following differences in the **FER2013 Dataset.**

1. **Low Resolution & Grayscale**  
   - Each image in *FER2013* is *48×48 pixels* and single-channel (grayscale).  
   - This severely **limits** the amount of *texture and color* information available for the model to learn from.

2. **Subtle Inter-Class Differences**  
   - Emotions such as *fear* vs. *surprise* or *sadness* vs. *neutral* often differ by only a few pixels along the eyebrows or mouth.  
   - Conventional object datasets (e.g., *CIFAR-10*) leverage *distinct shapes, colors, or contexts that are far more pronounced*.

3. **High Intra-Class Variability**  
   - A single emotion can manifest very differently across *age, gender, ethnicity, and individual facial structure*.  
   - Lighting conditions, head tilt, occlusions (e.g., glasses, hair), and background “noise” *further diversify examples within the same label.*
  
4. **Class Imbalance & Label Noise**  
   - Some emotions (e.g., *happy*) are over-represented, while others (e.g., *disgust*) are rare—leading to *imbalanced learning signals*.  
   - Human annotators often *disagree on subtle expressions*, introducing noise into the target labels.

5. **Limited Data Size for Deep Models**  
    - *FER2013* contains ~35,000 images—ample for classical machine-learning techniques but small for deep *convolutional* architectures prone to *overfitting*.  
   - Data augmentation *helps* but **cannot fully compensate for the lack of genuine diversity**.

6. **Lack of Contextual Cues**  
   - Emotions are often disambiguated by body language or situational context, neither of which is present in a tightly cropped face image.  
   - Models cannot rely on co-occurring objects (e.g., a birthday cake for “happy”) to guide predictions.

#### How can the challenges of the Fer2013 Dataset be overcome?

##### Implications for Model Training

1. **Architectural Considerations**  
   - **Deeper but Narrower Networks**  
     - To focus on very small features, it’s advantageous to use many layers with *smaller receptive fields (e.g., 3×3 kernels).*
   - **Attention Mechanisms**  
     - Spatial attention modules can help highlight relevant facial regions (eyes, mouth).

2. **Data Augmentation Strategies**  
   - **Geometric Transforms**  
     - Small rotations (±10°), shifts, and zooms simulate natural head movements.  
   - **Photometric Jitter**  
     - Subtle brightness/contrast adjustments mimic varied lighting.  
   - **Random Occlusion**  
     - “Cutout” patches train the model to be robust when parts of the face are hidden.

3. **Regularization & Transfer Learning**  
   - **Pretraining on Larger Face Datasets**  
     - Models can first learn general facial features (identity, pose) on datasets like VGGFace, then fine-tune on FER2013.  
   - **Dropout and Weight Decay**  
     - Essential to prevent overfitting given the small size and label noise.

4. **Handling Class Imbalance**  
   - **Weighted Loss Functions**  
     - Assign higher cost to under-represented classes to ensure their gradients meaningfully contribute.  
   - **Oversampling / Synthetic Samples**  
     - Generate additional examples of rare emotions via SMOTE-style interpolation in feature space.

By acknowledging these challenges—low resolution, subtle feature differences, subjective labels, and limited data, the newer V2 can tailor both our network design and training regimen to better suit the unique demands of facial expression recognition. 

#### Changes to V2

    Taking this into mind I changed multiple parts of my first version some