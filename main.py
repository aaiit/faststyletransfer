import streamlit as st
import cv2
import numpy as np

# from stylize import stylize
import torch
import utils
import transformer
import os
from torchvision import transforms
import time
import cv2

#transfrom image with proper style 
#Run on "CPU" only 

'''
Input:  
image (array):  The image need to be styled as np.array or cv2 format
style (string):  string represent style from trained model
        There are 5 style you passed in style
        -> bayanihan, lazy, mosaic, starry, tokyo_ghoul, udnie, wave
preserve_color (boolean): if True, keep the color distribution of the original image
                          if False, use the color distribution of given style

output_height (int): desired height of output image
output_width  (int): desired width of output image
Note that output_height and output_width also used to resize the input image
if both value are None, keep shape of the input image
if output_height is None, use output_width and infer output_height from the original h/w ratio
if output_width is None, use output_height and infer output_width from the original h/w ratio
if both value are not None, use output_width and infer output_height from the original h/w ratio

Output:
styled_image (array) styled image 
'''

def stylize(image, style, perserve_color = False, output_hight = None, output_width = None):

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(f"./pretrained_models/{style}.pth", map_location=torch.device('cpu')))
    net = net.to('cpu')


    with torch.no_grad():
        torch.cuda.empty_cache()
        print("Start styling image")

        content_image = image
        if output_width is not None:
            if image.shape[1] > output_width:
                content_image = utils.resize_image(image, width = output_width, height = output_hight)

        elif output_hight is not None:
            if image.shape[0] > output_hight:
                content_image = utils.resize_image(image, width = output_width, height = output_hight)




        starttime = time.time()
        content_tensor = utils.itot(content_image).to('cpu')
        generated_tensor = net(content_tensor)
        styled_image = utils.ttoi(generated_tensor.detach())
        if perserve_color:
            styled_image = utils.transfer_color(content_image, styled_image)
        print("Styling Time: {}".format(time.time() - starttime))
    
    return styled_image
#transfrom image with proper style 
# testimage_paths = [""] 
image_path = "/content/unnamed.jpg"
# Get all the images with exntension of "jpg" or "png"
# [testimage_paths.extend(list(Path(testdir).glob(f'*.{ext}'))) for ext in ['jpg', 'png']]

# Get all the available models
pretrained_models = ['bayanihan','lazy', 'mosaic', 'starry', 'tokyo_ghoul', 'udnie', 'wave'] 
style = pretrained_models[0]
output_path = "/content/ok2.png"



# image = utils.load_image(str(image_path))
# styled_image = stylize(image, style, output_width = 1080)


def main():
	# page_options = ["Fast style transfer",'Solution overview']
	# page_selection = st.sidebar.selectbox("Choose Option", page_options)

	# if page_options == "Fast style transfer":
	st.write('### Fast style transfer Demo from ***InData Labs***')
	uploaded_file = st.file_uploader("Choose a file")

	# if st.button("Run"):
	# 	print(uploaded_file)
	if uploaded_file is not None:
		# Convert the file to an opencv image.
	    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
	    opencv_image = cv2.imdecode(file_bytes, 1)
	    st.write(opencv_image.shape)
	    # Now do something with the image! For example, let's display it:
	    st.image(opencv_image, channels="BGR")
	    st.image(stylize(opencv_image, style, output_width = 1080), channels="BGR")

if __name__ == '__main__':
    main()
