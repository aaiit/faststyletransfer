import streamlit as st
import cv2
import numpy as np

# from stylize import stylize
import torch
import transformer
import utils
from torchvision import transforms
import os
import time
import cv2


# Load Transformer Network
# net = transformer.TransformerNetwork()
# net.load_state_dict(torch.load(f"./pretrained_models/{style}.pth", map_location=torch.device('cpu')))
# net = net.to('cpu')

pretrained_models = ['bayanihan','lazy', 'mosaic', 'starry', 'tokyo_ghoul', 'udnie', 'wave'] 

nets = {style:transformer.TransformerNetwork() for style in pretrained_models}
for style in pretrained_models:
	nets[style].load_state_dict(torch.load(f"./pretrained_models/{style}.pth", map_location=torch.device('cpu')))
	nets[style] = nets[style].to("cpu")

def stylize(image, style, perserve_color = False, output_hight = None, output_width = None):
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
		generated_tensor = nets[style](content_tensor)
		styled_image = utils.ttoi(generated_tensor.detach())
		if perserve_color:
			styled_image = utils.transfer_color(content_image, styled_image)
		print("Styling Time: {}".format(time.time() - starttime))
	
	return styled_image


	
def main():
	# page_options = ["Fast style transfer",'Solution overview']
	# page_selection = st.sidebar.selectbox("Choose Option", page_options)

	# if page_options == "Fast style transfer":
	st.write('### Fast style transfer Demo from ***InData Labs***')
	
	images = ["images/%s"%i  for i in os.listdir('images/')]
	indices = [i.split('.')[0]  for i in os.listdir('images/')]
	st.image(images, width=100, caption=indices)

	# Get all the available models
	style = st.selectbox("select style:",pretrained_models)

	uploaded_file = st.file_uploader("Choose a file")

	# if st.button("Run"):
	#	  print(uploaded_file)
	if uploaded_file is not None:
			file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
			opencv_image = cv2.imdecode(file_bytes, 1)
			# st.write(opencv_image.shape)
			# Now do something with the image! For example, let's display it:
			st.image(opencv_image, channels="BGR",caption = "Input image")
			y = stylize(opencv_image, style, output_width = 1080)
			# st.write(y.shape,y.max())
			cv2.imwrite("unnamed.jpg",y)
			st.image("unnamed.jpg",caption = "Generated image")

if __name__ == '__main__':
	main()
