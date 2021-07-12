import collections
import cv2 as cv
import numpy as np
import threading
import math 


_mu1 = threading.Lock()
_mu2 = threading.Lock()

FRAMES_PER_SECOND = 30
SWITCHING_FRAME = 20
RIGID_MECHANISM = 0
AFFINE_MECHANISM = 1
RIGID_SCALE = 1
	
CORNER_THRESHOLD = 10
CORNER_RETRACK_FRAMES = 20

MAX_CORNERS = 100
CORNER_QUALITY = 0.01
MINIMUM_DISTANCE = 20

SMOOTHING_RADIUS = 10
HORIZONTAL_BORDER_CROP = 20

prev_to_cur_transform = collections.deque()
new_prev_to_cur_transform = collections.deque()
Trajectory = collections.deque()

video_path = "hippo.mp4"
OUTPUT_VIDEO_NAME = video_path.split('.')[0]+"_Best_Output_Testing.mp4"
cap = cv.VideoCapture(video_path)
cap2 = cv.VideoCapture(video_path)
cap2.set(cv.CAP_PROP_POS_FRAMES,SMOOTHING_RADIUS)
max_frames=cap.get(cv.CAP_PROP_FRAME_COUNT)


def RigidorAffine(Rigid, partialAffine,prev_corner2):

	newPointsRigid=[]
	newPointspartialAffine=[]

	for i in range(len(prev_corner2)):
		_xR = Rigid[0, 0]*prev_corner2[i][0][0] + Rigid[0,1] * prev_corner2[i][0][1] + Rigid[0, 2]
		_yR = Rigid[1, 0] * prev_corner2[i][0][0] + Rigid[1, 1] * prev_corner2[i][0][1] + Rigid[1, 2]
		_wR = 1
		_xA = partialAffine[0, 0] * prev_corner2[i][0][0] + partialAffine[0, 1] * prev_corner2[i][0][1] + partialAffine[0, 2]
		_yA = partialAffine[1, 0] * prev_corner2[i][0][0] + partialAffine[1, 1] * prev_corner2[i][0][1] + partialAffine[1, 2]
		_wA = 1

		newPointsRigid.append((_xR,_yR))
		newPointspartialAffine.append((_xA,_yA))

	avgDiffRigid = 0;
	avgDiffAffine = 0;
	for i in range(len(prev_corner2)):
		avgDiffRigid += math.sqrt((pow((prev_corner2[i][0][0] - newPointsRigid[i][0]), 2) + pow((prev_corner2[i][0][1] - newPointsRigid[i][1]), 2)))
		avgDiffAffine += math.sqrt((pow((prev_corner2[i][0][0] - newPointspartialAffine[i][0]), 2) + pow((prev_corner2[i][0][1] - newPointspartialAffine[i][1]), 2)))

	avgDiffRigid /= len(prev_corner2)
	avgDiffAffine /= len(prev_corner2)

	if (avgDiffRigid > avgDiffAffine):
		return AFFINE_MECHANISM

	elif (avgDiffRigid < avgDiffAffine):
		return RIGID_MECHANISM
	else:
		return RIGID_MECHANISM;


def motionEstimation():
	global new_prev_to_cur_transform
	global prev_to_cur_transform
	global max_frames
	global cap
	global cap2

	prev_corner=[]
	cur_corner=[]
	prev_corner2=[]
	cur_corner2=[]
	countRigidTransform = 0
	countAffineTransform = 0
	counter = 0;
	k=1
	currentMethod = "RIGID"
	I=np.matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
	last_T=I
	
	ret, prev = cap.read()
	prev_grey = cv.cvtColor(prev,cv.COLOR_BGR2GRAY)

	
	while(k!=max_frames):
		ret,cur=cap.read()
		if(ret == False):
			break

		cur_grey=cv.cvtColor(cur, cv.COLOR_BGR2GRAY)
		prev_grey_roi = prev_grey[10:prev_grey.shape[0]-20, 10:prev_grey.shape[1]-20]
		cur_grey_roi = cur_grey[10:prev_grey.shape[0]-20, 10:prev_grey.shape[1]-20]

		if (k % CORNER_RETRACK_FRAMES == 0):
			prev_corner=cv.goodFeaturesToTrack(prev_grey_roi, MAX_CORNERS, CORNER_QUALITY, MINIMUM_DISTANCE)
	
		if (len(prev_corner) > CORNER_THRESHOLD):
			cur_corner, status, err=cv.calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner,None)

			for i in range(len(status)):
				if (status[i]):
					prev_corner2.append(prev_corner[i])
					cur_corner2.append(cur_corner[i])
					
			try:
				if(currentMethod=="RIGID"):
					R,_ = cv.estimateAffinePartial2D(prev_corner,cur_corner)
				elif(currentMethod=="AFFINE"):
					pA,_ = cv.estimateAffine2D(prev_corner, cur_corner)
				prev_corner=cur_corner
				counter+=1
			except:
				R = I
				pA = I
				currentMethod = "RIGID"
				prev_corner=cv.goodFeaturesToTrack(cur_grey_roi, MAX_CORNERS, CORNER_QUALITY, MINIMUM_DISTANCE)
				print("R or pA is invalid")

			if (k % SWITCHING_FRAME == 0):
				R, _ = cv.estimateAffinePartial2D(np.float32(prev_corner2), np.float32(cur_corner2))
				pA, _ = cv.estimateAffine2D(np.float32(prev_corner2), np.float32(cur_corner2))
				if(R.size>0 and pA.size>0 and RigidorAffine(R, pA, prev_corner2) == 1):
					currentMethod = "AFFINE"
					countAffineTransform=countAffineTransform+1
				else:
					currentMethod = "RIGID"
					countRigidTransform=countRigidTransform+1
			
			if (currentMethod == "RIGID"):
				T = R
			elif (currentMethod == "AFFINE"):
				T = pA

			if (T.size == 0):
				T=last_T

			dx = T[0, 2]
			dy = T[1, 2]
			da = np.arctan2(T[1, 0], T[0, 0])

			if (dx < -100 or dx > 100):
				dx = 0
			else:
				last_T[0, 2]=T[0, 2]
			if (dy < -100 or dy > 100):
				dy = 0
			else:
				last_T[1, 2]=T[1, 2]
			if (da > 0.7853 or da < -0.7853):
				da = 0
			else:
				last_T=T


			if (currentMethod == "RIGID"):
				_mu1.acquire()
				prev_to_cur_transform.append((dx, dy, da, RIGID_SCALE))
				_mu1.release()
			elif (currentMethod == "AFFINE"):
				ds = T[0, 0] / np.cos(da)
				_mu1.acquire()
				prev_to_cur_transform.append((dx, dy, da, ds))
				_mu1.release()
		else:
			_mu1.acquire()
			prev_to_cur_transform.append((0, 0, 0, 1))
			_mu1.release()
			prev_corner=cv.goodFeaturesToTrack(cur_grey, MAX_CORNERS, CORNER_QUALITY, MINIMUM_DISTANCE)

		prev=cur
		prev_grey=cur_grey
		print(f"Frame: {k}/{max_frames} - good optical flow: {len(prev_corner2)}")
		k+=1
		prev_corner2=[]
		cur_corner2=[]

	print("Number of times RIGID transform takes place(per 20 frames): ",countRigidTransform)
	print("Number of times AFFFINE transform takes place(per 20 frames): ",countAffineTransform)
	print("counter: ",counter)



def motionCompensation():
	global new_prev_to_cur_transform
	global prev_to_cur_transform
	global Trajectory
	global max_frames
	global cap
	global cap2
	global temp
	k = 0
	while (True):
		if (len(prev_to_cur_transform) == SMOOTHING_RADIUS * 2):
			smoothed_Trajectory = collections.deque()

			_mu1.acquire()
			l=len(Trajectory)
			if(l==0):
				a = 0
				x = 0
				y = 0
				s = 0
				for i in range(SMOOTHING_RADIUS*2):
					x += prev_to_cur_transform[i][0]
					y += prev_to_cur_transform[i][1]
					a += prev_to_cur_transform[i][2]
					s += prev_to_cur_transform[i][3]
					Trajectory.append((x, y, a, s))
			elif(l==SMOOTHING_RADIUS*2):
				P=(Trajectory[l-1][0]+prev_to_cur_transform[l-1][0],Trajectory[l-1][1]+prev_to_cur_transform[l-1][1],Trajectory[l-1][2]+prev_to_cur_transform[l-1][2],Trajectory[l-1][3]+prev_to_cur_transform[l-1][3])
				Trajectory.popleft()
				Trajectory.append(P)
			
			sum_x = 0
			sum_y = 0
			sum_a = 0
			sum_s = 0
			for i in range(len(Trajectory)):
				sum_x += Trajectory[i][0]
				sum_y += Trajectory[i][1]
				sum_a += Trajectory[i][2]
				sum_s += Trajectory[i][3]
			avg_a = sum_a / (2*SMOOTHING_RADIUS)
			avg_x = sum_x / (2*SMOOTHING_RADIUS)
			avg_y = sum_y / (2*SMOOTHING_RADIUS)
			avg_s = sum_s / (2*SMOOTHING_RADIUS)

			diff_x = avg_x - Trajectory[SMOOTHING_RADIUS][0]
			diff_y = avg_y - Trajectory[SMOOTHING_RADIUS][1]
			diff_a = avg_a - Trajectory[SMOOTHING_RADIUS][2]
			diff_s = avg_s - Trajectory[SMOOTHING_RADIUS][3]

			dx = prev_to_cur_transform[SMOOTHING_RADIUS][0] + diff_x
			dy = prev_to_cur_transform[SMOOTHING_RADIUS][1] + diff_y
			da = prev_to_cur_transform[SMOOTHING_RADIUS][2] + diff_a
			ds = prev_to_cur_transform[SMOOTHING_RADIUS][3] + diff_s

			if (ds < 0.9 or ds > 1.4):
				ds = 1

			_mu2.acquire()
			new_prev_to_cur_transform.append((dx, dy, da, ds)) 
			_mu2.release()
			prev_to_cur_transform.popleft()
			_mu1.release()
		
		



def display():
	global new_prev_to_cur_transform
	global prev_to_cur_transform
	global max_frames
	global cap
	global cap2
	#fourcc = cv.VideoWriter_fourcc(*'mp4v')
	#result=cv.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 30, (540 * 2 + 10, 360))
	cap2.set(cv.CAP_PROP_POS_FRAMES, SMOOTHING_RADIUS)
	T = np.zeros((2,3), np.float32)
	k=0
	while (True):
		if (len(new_prev_to_cur_transform)>0):
			ret, cur = cap2.read()
			if(ret==False):
				break
			
			vert_border = int(HORIZONTAL_BORDER_CROP * cur.shape[0] / cur.shape[1])
			_mu2.acquire()
			P = new_prev_to_cur_transform.popleft()
			_mu2.release()

			if (P[3] != RIGID_SCALE):
				T[0, 0] = P[3] * np.cos(P[2])
				T[0, 1] = P[3] * -np.sin(P[2])
				T[1, 0] = P[3] * np.sin(P[2])
				T[1, 1] = P[3] * np.cos(P[2])
			else:
				T[0, 0] = np.cos(P[2])
				T[0, 1] = -np.sin(P[2])
				T[1, 0] = np.sin(P[2])
				T[1, 1] = np.cos(P[2])

			T[0, 2] = P[0]
			T[1, 2] = P[1]
			cur2=cv.warpAffine(cur, T, (cur.shape[1],cur.shape[0]))

			cur2 = cur2[vert_border:cur2.shape[0] - vert_border, HORIZONTAL_BORDER_CROP: cur2.shape[1] - HORIZONTAL_BORDER_CROP]

			cur2 = cv.resize(cur2,(cur.shape[1],cur.shape[0]))

			canvas = np.zeros([cur.shape[0], cur.shape[1] * 2 + 10,3], dtype="uint8")

			canvas[:,0:cur2.shape[1]] = cur
			canvas[:,cur2.shape[1]+10:cur2.shape[1] *2 +10] =cur2
			if (canvas.shape[1] > 1920):
				canvas=cv.resize(canvas, (canvas.shape[1] / 2, canvas.shape[0] / 2))
			#if(k<=400):
			#	result.write(canvas)
			#else:
			#	result.release()
			k+=1
			cv.imshow("before and after", canvas)
			cv.waitKey(30);
	#result.release()
			



if __name__ == "__main__":

	t1 = threading.Thread(target=motionEstimation)
	t2 = threading.Thread(target=motionCompensation)
	t3 = threading.Thread(target=display)
	t1.start()
	t2.start()
	t3.start()
	t1.join()
	t2.join()
	t3.join()
	cap.release()
	cap2.release()
	cv.destroyAllWindows()

	
