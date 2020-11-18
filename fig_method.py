import numpy as np
import math
from scipy import signal

#  probe data analysis

class data_cal:
	@staticmethod
	def font_standard(self):
		pass
	def merge_fig(self):
		pass

	@classmethod
	def __init__(self, Nwindow = 8, noverlap = 0.5, nfft = 0, fs = 1):
		self.Nwindow = Nwindow;
		self.noverlap = noverlap;
		self.nfft = nfft;
		self.fs = fs;

	def disp(self):
		print('Nwindow = {}\nnoverlap = {}\nnfft = {}\nfs = {}'.format(
			self.Nwindow, self.noverlap, self.nfft, self.fs))

	def intervalcut(self, sig):
		nover = self.noverlap;
		Nwindows = self.Nwindow;

		len_sig = len(sig);
		len_win = math.floor(len_sig / (Nwindows - (Nwindows-1)*nover));
		len_nover = math.floor(nover * len_win);
		Nstart = np.zeros(Nwindows, dtype = 'int64');
		Nend = np.zeros(Nwindows, dtype = 'int64');
		t = np.zeros(Nwindows);

		Nstart[0] = 0;
		Nend[0] = len_win - 1;
		t[0] = (1+len_win)/2;
		for N1 in range(1, Nwindows):
			Nstart[N1] = Nend[N1-1] - len_nover + 1;
			Nend[N1] = Nstart[N1] + len_win - 1;
			t[N1] = (Nstart[N1] + Nend[N1])/2;
		Nend = Nend + np.ones(Nwindows, dtype = 'int64');
		if Nend[-1] > len_sig - 1:
			Nend[-1] = len_sig - 1;
		return Nstart, Nend, t;

	def cross_coeff(self, sig1, sig2):
		nfft = self.nfft;
		if nfft == 0:
			nfft = None;
		if len(sig1) != len(sig2):
			print('array lengths must be equal')
			return;
		intval_start, intval_end, t = self.intervalcut(sig1);
		t = t/self.fs;
		temp1, f = signal.coherence(sig1[intval_start[0]:intval_end[0]] 
				, sig2[intval_start[0]:intval_end[0]] , fs = self.fs, nfft = nfft);
		data = np.zeros((len(f),self.Nwindow))
		for N1 in range(self.Nwindow):
			temp1, f = signal.coherence(sig1[intval_start[N1]:intval_end[N1]] 
				, sig2[intval_start[N1]:intval_end[N1]] , fs = self.fs, nfft = nfft);
			data[:,N1] = temp1;
		return data, t, f;

def timeslice(t, start_t, end_t):
	# start_time = datetime.datetime.now()
	low = 0;
	high = len(t) - 1;
	while low <= high:
		mid = (low + high) // 2;
		if t[mid] < start_t:
			low = mid + 1;
		elif t[mid] > end_t:
			high = mid - 1;
		else:
			break;
	low1 = low; 
	high1 = (low + high) // 2;
	low2 = high1 + 1;
	high2 = high;
	while low1 <= high1:
		mid = (low1 + high1) // 2;
		if t[mid] < start_t:
			low1 = mid + 1;
		elif t[mid] > start_t:
			high1 = mid - 1;
		else:
			start_index = mid;
			break;
	start_index = mid;

	while low2 <= high2:
		mid = (low2 + high2) // 2;
		if t[mid] < end_t:
			low2 = mid + 1;
		elif t[mid] > end_t:
			high2 = mid - 1;
		else:
			end_index = mid;
			break;
	end_index = mid;
	# print(datetime.datetime.now() - start_time);
	return start_index, end_index;

def timeslice_slow(t, start_t, end_t):
	# start_time = datetime.datetime.now()
	k = 0;
	while k < len(t) - 1:
		if start_t > t[k]:
			k = k + 1;
		else: 
			start_index = k;
			break;
	while k < len(t) - 1:
		if end_t > t[k]:
			k = k + 1;
		else:
			end_index = k;
			break;
	# print(datetime.datetime.now() - start_time);
	return start_index, end_index;
            

# demo
if __name__ == '__main__':
	example = fig_method();
	example.disp();
	start,end,t = example.intervalcut(np.linspace(1,160,160));
	data,t,f = example.cross_coeff(np.linspace(1,2,512),np.linspace(1,2,512));