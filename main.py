import streamlit as st
import cv2
import numpy as np

from stylized import stylized
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

	    st.image(stylized(opencv_image), channels="BGR")
		# image = uploaded_file.read()
		# im1=cv2.imread(image)
		# st.write(im)
		# # img = st.image(image, caption='Original image', use_column_width=True)
		# # ok = cv2.imread(img.name)
		# st.write(ok.shape)

		# img = st.image(image, caption='Generated image', use_column_width=True)

if __name__ == '__main__':
    main()
