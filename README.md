# UAV-GESTURE
UAV-GESTURE: A Dataset for UAV Control and Gesture Recognition

This is a slightly modified version of the original P-CNN implementation (https://github.com/gcheron/P-CNN).

SVM Classification files are from https://github.com/ZhigangTU/HR-MSCNN.

How to run the code. 

1. Follow the original installation instructions availabe on https://github.com/gcheron/P-CNN

2. Download the RGB and FLOW networks from  here (https://www.di.ens.fr/willow/research/p-cnn/download/models.tar), and save them in the "models" folder.

3. Extract images using "video_to_frames.m" file.

4. Run "demo.m". Update "param.impath" variable with extracted images path.

5. Estimated body joints (using OpenPose) are already saved in 'UAVGESTURE/joint_positions'.

6. Please send an email to asanka.perera@mymail.unisa.edu.au to get the video dataset download link.

7. Project website: https://asankagp.github.io/uavgesture/
