# NCC+LSM in Digital Photogrammetry

## Image Matching Based on Normalized Correlation Coefficient (NCC) and Least Squares Matching (LSM)

### NCC+LSM.cpp:
NCC is realized based on Moravec operator and Harris operator and LSM Refines Matching Results.

1.OpenCV 4.5.0 is used.

2.Chinese paths are prohibited.

3.Results (.jpg&.txt) are saved in the path of the second image.

4.Default parameters:(Line 60~77)

		//moravec params
		par1_m.winSize = 9;
		par1_m.threshold = 8000;
		par1_m.restrainWinSize = 80;

		//harris params
		params_harris par1_h;
		par1_h.blockSize = 4;
		par1_h.apertureSize = 5;
		par1_h.rc = 0.05;
		par1_h.thHarrisRes = 130;

		//ncc params
		par2.matchsize = 9;
		par2.PreSearchRadius = 15;
		par2.dist_width = 767;
		par2.dist_height = 0;
		par2.lowst_door = 0.85;
### l/r.jpg:
A pair of aerial photos. (left & right) 
