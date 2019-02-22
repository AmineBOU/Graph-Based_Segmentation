#!/usr/bin/python3
'''
	Copyright 2019 Mohammed Azizi & Mohamed Amine BOURKHISS 

	Graph-based segmentation method.

	For the sake of simplicity SP indicates super-pixel :)
'''

# Needed packages
import sys, getopt
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import boykov_kolmogorov

# Init object params
drawing = False
mode    = "ob"

marked_ob_pixels = []
marked_bg_pixels = []

I        = None
I_dummy  = None
l_range  = [0,256]
a_range  = [0,256]
b_range  = [0,256]
lab_bins = [32,32,32]

# Define SPNode class
class SPNode():
	"""
	SPNode class to characterize super-pixels. Characteristics are : 
	-- label : "bg" or "ob"
	-- pixels : SP's pixels
	-- mean_intensity 
	-- centroid : SP's centroid
	-- type 
	-- mean_lab  
	-- lab_hist 
	-- real_lab  
	"""
	def __init__(self):
		self.label  = None
		self.pixels = []
		self.mean_intensity = 0.0
		self.centroid = ()
		self.type = 'na'
		self.mean_lab = None
		self.lab_hist = None
		self.real_lab = None

	def __repr__(self):
		return str(self.label)

# Mark seeds
def mark_seeds(event, x, y, flags, param):
	"""
	Function to mark seeds to identify background and object 
	"""
	global drawing, mode, marked_bg_pixels, marked_ob_pixels, I_dummy
	h, w, c = I_dummy.shape

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == "ob":
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_ob_pixels.append((y,x))
				cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
			else:
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_bg_pixels.append((y,x))
				cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == "ob":
			cv2.line(I_dummy,(x-3,y),(x+3,y),(0,0,255))
		else:
			cv2.line(I_dummy,(x-3,y),(x+3,y),(255,0,0))
	
# Function to generate SP seeds
def gen_sp_seed(I, h, w, c):
	# Superpixel Generation :: Superpixels extracted via energy-driven sampling
	num_sp   = 500
	num_iter = 4
	num_block_levels = 1

	sp_seeds = cv2.ximgproc.createSuperpixelSEEDS(w, h, c, num_sp, num_block_levels, prior = 2, histogram_bins=5, double_step = False)
	sp_seeds.iterate(I,num_iterations=num_iter)
	
	return sp_seeds

# Function to generate superpixels
def gen_sp_slic(I, region_size_):
	# Superpixel Generation ::  Slic superpixels compared to state-of-the-art superpixel methods
	SLIC=100
	SLICO=101
	num_iter=4
	sp_slic=cv2.ximgproc.createSuperpixelSLIC(I, algorithm=SLICO, region_size=region_size_, ruler=10.0)
	sp_slic.iterate(num_iterations=num_iter)

	return sp_slic

# Function to draw sp masks
def draw_sp_mask(I, SP):
	I_marked = np.zeros(I.shape)
	I_marked = np.copy(I)
	mask = SP.getLabelContourMask()
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i][j]==-1 or mask[i][j]==255: # SLIC/SLICO marks borders with -1 :: SEED marks borders with 255
				I_marked[i][j]=[128,128,128]
	return I_marked

# Function to draw sp centroids
def draw_centroids(I, SP_list):
	for each in SP_list:
		i, j    = each.centroid
		I[i][j] = 128
	return I

# Function to draw graph edges
def draw_edges(I, G):
	for each in G.edges():
		cv2.line(I, each[0].centroid[::-1], each[1].centroid[::-1], 128)
	return I

# Compute 2D euclidean distance
def distance(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# Compute 3D euclidean distance
def distance_3d(p0, p1):
	return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2]-p1[2])**2)

# Augmenter la luminosité d'une image
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

# Générer un graphe d'image
def gen_graph(I, SP_list, hist_ob, hist_bg):
	G = nx.Graph()
	s, t = SPNode(), SPNode()
	s.label, t.label = 's', 't'

	lambda_, sig_ = .9, 5

	hist_ob_sum = int(hist_ob.sum())
	hist_bg_sum = int(hist_bg.sum())

	for u in SP_list:
		K = 0
		region_rad = math.sqrt(len(u.pixels)/math.pi)
		for v in SP_list:
			if u != v:
				if distance(u.centroid, v.centroid) <= 2.5*region_rad:
					sim = math.exp(-(cv2.compareHist(u.lab_hist,v.lab_hist,3)**2/2*sig_**2))*(1/distance(u.centroid, v.centroid))
					K  += sim
					G.add_edge(u, v, sim = sim)

		if(u.type=='na'):
			l_,a_,b_=[int(x) for x in u.mean_lab]
			l_i=int(l_//((l_range[1]-l_range[0])/lab_bins[0]))
			a_i=int(a_//((a_range[1]-a_range[0])/lab_bins[1]))
			b_i=int(b_//((b_range[1]-b_range[0])/lab_bins[2]))
			pr_ob=int(hist_ob[l_i,a_i,b_i])/hist_ob_sum
			pr_bg=int(hist_bg[l_i,a_i,b_i])/hist_bg_sum
			sim_s=1000
			sim_t=1000
			if pr_bg > 0:
				sim_s=lambda_*-np.log(pr_bg)
			if pr_ob > 0:
				sim_t=lambda_*-np.log(pr_ob)
			G.add_edge(s, u, sim=sim_s)
			G.add_edge(t, u, sim=sim_t)
		
		if(u.type=='ob'):
			G.add_edge(s, u, sim=1+K)
			G.add_edge(t, u, sim=0)

		if(u.type=='bg'):
			G.add_edge(s, u, sim=0)
			G.add_edge(t, u, sim=1+K)		
	return G


#-------------------------------------------------------------------------#
def main():
	global I, mode, I_dummy

	# Parse image's fillename
	inputfile = ''
	try:
		opts, args = getopt.getopt(sys.argv[1:], "i:h", ["input-image=", "help"])
	except getopt.GetoptError:
		print('fast_seg.py -i <input image>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('fast_seg.py -i <input image>')
			sys.exit()
		elif opt in ("-i", "--input-image"):
			inputfile = arg
	print('Using image: ', inputfile)

	# Read and copy image
	I       = cv2.imread(inputfile)
	I       = increase_brightness(I, value=10) # Modify img'sSP brightness 
	I_dummy = np.zeros(I.shape)
	I_dummy = np.copy(I)
	
	# Retrieve the shape of img
	h,w,c       = I.shape
	region_size = 20

	# Create a window (user interface)
	cv2.namedWindow('Mark the object and background')
	# Set mouse handler for the specified window
	cv2.setMouseCallback('Mark the object and background', mark_seeds)

	# Mark the object and the background while as long as the user does not press "esc"
	# To mark the object     : the user could press "o"
	# To mark the background : the user should press "b"
	# Press "esc" when marking seeds is done
	while(1):
		cv2.imshow('Mark the object and background',I_dummy)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('o'):
			mode = "ob"
		elif k == ord('b'):
			mode = "bg"
		elif k == 27:
			break
	cv2.destroyAllWindows()
	
	# 
	I_lab = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
	SP    = gen_sp_slic(I, region_size)

	# Init SP' labels and list
	# SP_labels stores SP representation which is the corresponding label for each SP
	# SP_list stores SP objects
	SP_labels = SP.getLabels()
	SP_list   = [None for each in range(SP.getNumberOfSuperpixels())]

	# Construct SP_list
	for i in range(h):
		for j in range(w):
			if not SP_list[SP_labels[i][j]]:
				tmp_sp       = SPNode()
				tmp_sp.label = SP_labels[i][j]
				tmp_sp.pixels.append((i,j))
				SP_list[SP_labels[i][j]] = tmp_sp
			else:
				SP_list[SP_labels[i][j]].pixels.append((i,j))

	# Compute SP node attributes
	for sp in SP_list:
		n_pixels = len(sp.pixels)
		i_sum, j_sum = 0, 0
		lab_sum  = [0,0,0]
		tmp_mask = np.zeros((h,w),np.uint8)
		
		for each in sp.pixels:
			i, j   = each
			i_sum +=i
			j_sum +=j

			lab_sum = [x + y for x, y in zip(lab_sum, I_lab[i][j])]
			tmp_mask[i][j] = 255

		sp.lab_hist  = cv2.calcHist([I_lab],[0,1,2],tmp_mask,lab_bins,l_range+a_range+b_range)
		sp.centroid += (i_sum//n_pixels, j_sum//n_pixels,)
		sp.mean_lab  = [x/n_pixels for x in lab_sum]
		sp.real_lab  = [sp.mean_lab[0]*100/255,sp.mean_lab[1]-128,sp.mean_lab[2]-128]

	# Define SPNode type marked by the user as "ob" (for object) 
	for pixels in marked_ob_pixels:
		x,y = pixels
		SP_list[SP_labels[x][y]].type="ob"
	
	# Define SPNode type marked by the user as "bg" (for background) 
	for pixels in marked_bg_pixels:
		x,y = pixels
		SP_list[SP_labels[x][y]].type="bg"

	# Construct the graph by drawing superpixels mask and centroids 
	I_marked=draw_sp_mask(I,SP)
	I_marked=draw_centroids(I_marked,SP_list)
	
	# Define object mask
	mask_ob=np.zeros((h,w),dtype=np.uint8)
	for pixels in marked_ob_pixels:
		i, j = pixels
		mask_ob[i][j] = 255

	# Define background mask
	mask_bg = np.zeros((h,w),dtype=np.uint8)
	for pixels in marked_bg_pixels:
		i,j=pixels
		mask_bg[i][j]=255


	# Compute object and background histograms
	hist_ob = cv2.calcHist([I_lab],[0,1,2],mask_ob,lab_bins,l_range+a_range+b_range)
	hist_bg = cv2.calcHist([I_lab],[0,1,2],mask_bg,lab_bins,l_range+a_range+b_range)

	# Generate a NetworkX graph
	G = gen_graph(I_lab, SP_list, hist_ob, hist_bg)

	# Cluster G nodes
	for each in G.nodes():
		if each.label == 's':
			s = each
		if each.label == 't':
			t = each

	# Compute max-flow using boykov_kolmogorov algo
	RG = boykov_kolmogorov.boykov_kolmogorov(G, s, t, capacity='sim')

	# Compute source (S) and target (T) sets
	source_tree, target_tree = RG.graph['trees']
	partition = (set(source_tree), set(G) - set(source_tree))
	
	# Generate labels matrix for binary segmentation
	F = np.zeros((h,w),dtype = np.uint8)
	for sp in partition[0]:
		for pixels in sp.pixels:
			i, j    = pixels
			F[i][j] = 1

	# Compute the per-element bit-wise conjunction of img and itself having F as a mask		
	Final = cv2.bitwise_and(I, I, mask = F)

	# Compute SP representation
	sp_lab = np.zeros(I.shape, dtype = np.uint8)
	for sp in SP_list:
		for pixels in sp.pixels:
			i, j = pixels
			sp_lab[i][j] = sp.mean_lab
	sp_lab = cv2.cvtColor(sp_lab, cv2.COLOR_Lab2RGB)
	
	plt.figure(figsize=(16, 26))
	plt.subplot(2,2,1)
	plt.tick_params(labelcolor='black', top='off', bottom='off', left='off', right='off')
	plt.imshow(I[...,::-1])
	plt.axis("off")
	plt.title("Input image")
	plt.xlabel("Input image")
	
	plt.subplot(2,2,2)
	plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
	plt.imshow(I_marked[...,::-1])
	plt.axis("off")
	plt.title("Super-pixel boundaries and centroid")
	plt.xlabel("Super-pixel boundaries and centroid")

	plt.subplot(2,2,3)
	plt.imshow(sp_lab)
	plt.axis("off")
	plt.title("Super-pixel representation")
	plt.xlabel("Super-pixel representation")

	plt.subplot(2,2,4)
	plt.imshow(Final[...,::-1])
	plt.axis("off")
	plt.title("Output Image")
	plt.xlabel("Output Image")


	
	cv2.imwrite("out.png", Final)
	cv2.imwrite("bright.png", I)

	plt.show()

if __name__ == '__main__':
	main()