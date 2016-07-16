import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import time
import warnings;
with warnings.catch_warnings():
    warnings.simplefilter("ignore");
    import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def knightMoves(cpos):
	if cpos == 1:
		allowed = [3,8]
	elif	cpos == 2:
		allowed = [7,9]
	elif cpos == 3:
		allowed = [4,8]
	elif cpos == 4:
		allowed = [3,9,0]
	elif cpos == 6:
		allowed = [1,7,0]
	elif cpos == 7:
		allowed = [2,6]
	elif cpos == 8:
		allowed = [1,3]
	elif cpos == 9:
		allowed = [2,4]
	elif cpos == 0:
		allowed = [4,6]

	newpos = random.choice(allowed)
	return newpos
	
def plotHistofSum(T,data,binwidth):
	#fit a normal distribution to the data
	mu, std = norm.fit(data)
	
	#plot histogram
	plt.hist(data,bins=range(min(data),max(data)+binwidth,binwidth))
	
	#plot Prob. Density Function
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin,xmax,100)
	p = norm.pdf(x,mu,std)
	plt.plot(x,p,'k',linewidth=2)
	title = 'Fit results: mu = %.2f, std = %.2f' % (mu,std)
	plt.title(title)
	plt.show()

def plotHistofModSum(T,data,binwidth):
	#fit a normal distribution to the data
	mu, std = norm.fit(data)
	
	#plot histogram
	plt.hist(data,bins=range(min(data),max(data)+binwidth,binwidth))
	
	#plot Prob. Density Function
	xmin, xmax = plt.xlim()
	x = np.linspace(xmin,xmax,100)
	p = norm.pdf(x,mu,std)
	plt.plot(x,p,'k',linewidth=2)
	title = 'Fit results: mu = %.2f, std = %.2f' % (mu,std)
	plt.title(title)
	plt.show()

#Main script
K = 0
S = 0
probT10 = 0;
probT1024 = 0;
T = int(raw_input('What is T?'))
N = int(raw_input('How many times you want to run (N)?')) #number of times to run script
values_S = []
modS = []

f = open('data5','w')
k = open('Sum_data5','w')

for i in range(0,N):
	S, K = 0, 0
	for n in range(T):
		K = knightMoves(K)
		S = S + K
	
	values_S.append(S)
	modS.append(S%T)
	
	f.write('%.12f %.12f\n' % (S, S%T))
	
	if T == 10:
		if S%7 == 0 and S%5==0:
			probT10 += 1
	elif T == 1024:
		if S%29 == 0 and S%23 == 0:
			probT1024 += 1
					
	#print('The sum S of the chess piece knight: {}').format(values_S[i])
	#print('The S mod {} of the chess piece knight: {}').format(T,modS[i])

k.write('%i %.12f %.12f %.12f' % (N, np.sum(values_S),np.sum(modS),np.var(modS)))
f.close()
k.close()

if T == 10:
	print 'number of times div by 5 and 7: {}'.format(probT10)
	probability = float(probT10)/float(N)	
	print 'Probability T = 10 and div by 5 and 7: {}'.format('%.12f' % probability)
elif T == 1024:
	print 'number of times div by 29 and 23: {}'.format(probT1024)
	probability = float(probT1024)/N	
	print 'Probability T = 1024 and div by 23  and 29: {}'.format('%.12f' % probability)	
	
	
mean =np.mean(modS)
stddev = np.std(modS)
print 'Mean value of S mod {}: {}'.format(T, '%.12f' % mean)
print 'Standard Deviation of S mod {}: {}'.format(T, '%.12f' % stddev)

#PLOT HISTOGRAM
#plotHistofSum(T,values_S,2)
#plotHistofModSum(T,modS,1)

		