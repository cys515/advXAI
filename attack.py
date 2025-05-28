import numpy as np
import time
from torch.autograd import Variable
import torch
import scipy
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from explain import getAttribution
import multiprocessing
from scipy.optimize import approx_fprime, differential_evolution
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from smooth import get_smooth_saliency

class Attack(object):
	
	def __init__(self, args, test, label, model, saliency, saliency_method, device, k_top=10):
		self.test = test	#tensor [1,t,f]
		self.shape = test.shape
		self.original_label = label.reshape(label.shape[0],1).to(device)	#tensor [num,1]
		self.model = model
		self.saliency = torch.from_numpy(saliency).to(device)	#tensor [t,f]
		self.test_flatten = self.test.reshape(-1)		#tensor [t*f]
		self.saliency_flatten = self.saliency.reshape(-1)	#tensor [t*f]
		self.saliency_method = saliency_method

		self.k_top = k_top
		self.topK = torch.argsort(self.saliency_flatten)[-self.k_top:]	#sort from small to large 
		self.mass_center = self.cal_mass_center(self.saliency,device)


	def cal_mass_center(self, saliency,device):
		# t,f = saliency.shape
		# y_mesh, x_mesh = np.meshgrid(np.arange(f), np.arange(t))
		# mass_center = np.stack([np.sum(saliency*x_mesh)/(t*f),np.sum(saliency*y_mesh)/(t*f)])
		t,f = saliency.shape
		y_mesh, x_mesh = torch.meshgrid(torch.arange(t), torch.arange(f), indexing='ij')
		y_mesh, x_mesh = y_mesh.to(device), x_mesh.to(device)
		mass_center = torch.stack([torch.sum(saliency*x_mesh)/(t*f),torch.sum(saliency*y_mesh)/(t*f)])
		return mass_center

	def predict(self, input):
		self.model.eval()
		outputs = self.model(input)
		return outputs
		
	def attack(self, attack_method, metric, device, alpha, iters=100):
		"""
		create final adversial sample to a original sample
		Args:
			attack_method: [PSO | GA]
			metric: One of "mass_center", "topK" or "random"
			epsilon: Allowed maximum $ell_infty$ of perturbations, eg:8
			iters: number of maximum allowed attack iterations
			alpha: perturbation size in each iteration of the attack
		Returns:
			samples: perturbed samples x^p
			explanation: perturbed explanation I(x^p)
			criterion: The difference between origin explanation and perturbed explanation
			perturb_size: The distance of perturbed sample and original sample
		""" 
		if torch.any(torch.isnan(self.saliency)) == True:
			return self.test.cpu().numpy(), self.saliency.cpu().numpy()

		#only optimize part points
		# tmp = torch.full((self.topK.shape),alpha).to(device)
		# part = torch.argsort(self.saliency_flatten)[-self.k_top:]	#top K time points location
		# part = torch.argsort(self.saliency_flatten)[:self.k_top]		#last K time points
		
		#optimize full points	
		tmp = torch.full((self.test_flatten.shape),alpha).to(device)
		part = torch.argsort(self.saliency_flatten)

		opti_sample = self.test_flatten[part] 		#tensor [k_top]
		bounds = torch.stack([opti_sample-tmp,opti_sample+tmp],dim=1)  #[t*f,2]

		if attack_method == "PSO":
			sample,criterion = self.PSO(metric, bounds, part, device, iters)
		
		if attack_method == "GA":			
			sample, criterion = self.GA(metric, bounds, device, iters, popsize=10)

		if attack_method == "PGD":
			sample = self.PGD(metric, bounds, device, iters, step = 0.005)
		sample = sample.reshape(self.shape)
		label = self.predict(sample)
		attr = getAttribution(sample, label, self.model, self.saliency_method)
		# attr = get_smooth_saliency(sample[0], self.shape[1], self.shape[2], device, 50, "DBA", self.model, self.saliency_method)
		# return sample.cpu().numpy(), attr.cpu().numpy()
		return sample.detach().cpu().numpy(), attr.detach().cpu().numpy()


	def GA(self, metric, bounds, device, iters, popsize, mut=0.8, crossp=0.7):

		adv_sample = self.test_flatten.repeat(popsize,1)
		sample = self.test_flatten.clone()

		d = bounds.shape[0]  #Individual dimension
		pop = torch.rand(popsize, d).to(device)	#random generate population
		min_b, max_b = bounds.t()		#get upper and lower bound
		diff = torch.abs(min_b - max_b)	
		pop_denorm = min_b + pop * diff   #scale population

		topK_expanded = self.topK.unsqueeze(0).expand(adv_sample.size(0), -1)
		adv_sample[:, topK_expanded[0]] = pop_denorm

		fitness = self.obj_func(adv_sample, metric, device)
		best_idx = torch.argmin(fitness)	#find best ind
		best = adv_sample[best_idx]		#save best ind

		#iterate optimize
		for i in range(iters):
			for j in range(popsize):
				idxs = [idx for idx in range(popsize) if idx != j]
				a, b, c = pop[np.random.choice(idxs, 3, replace=False)]		#random select 3 ind except current ind
				mutant = torch.clamp(a + mut * (b - c), 0, 1)  #Mutation

				cross_points = torch.rand(d).to(device) < crossp		#Gene crossing
				if not torch.any(cross_points):
					cross_points[torch.randint(0, d, (1,))] = True
				trial = torch.where(cross_points, mutant, pop[j])		#true->mutant, false->keep
				trial_denorm = min_b + trial * diff	

				sample[self.topK] = trial_denorm
				f = self.obj_func(sample.unsqueeze(0), metric, device)

				if f < fitness[j]:		#evolution
					fitness[j] = f
					pop[j] = trial
					if f < fitness[best_idx]:
						best_idx = j
						best = sample

		return best, fitness[best_idx]

	def PSO(self, metric, bounds, part, device, iterations=100, num_particles=40, w=1.0, c1=0.6, c2=0.6):
		
		d = bounds.shape[0]
		adv_sample = self.test_flatten.repeat(num_particles,1)  #tensor [num_particles, t*f] whole sample
		sample = self.test_flatten.clone()

		#initial particle location and speed
		particles = self.test_flatten[part].repeat(num_particles,1)	#[num_particles, k_top] part time points value
		speed = torch.rand(num_particles, d).to(device)		#[num_particles, k_top]
		
		#initial particle best location and global best location
		best_pos = particles.clone()
		best_val = torch.ones(num_particles).to(device)
		global_best_pos = particles[0].clone()   #[d]
		global_best_val = 1

		#iteration optimize
		for i in range(iterations): 
			#update particle location and speed
			r1,r2 = torch.rand(2,num_particles,d).to(device)  #[num_particles,d]
			speed = w * speed + c1*r1*(best_pos-particles) + c2*r2*(global_best_pos-particles)
			particles += speed
			particles = torch.min(torch.max(particles,bounds[:,0]),bounds[:,1])	#clip
					
			adv_sample[:, part] = particles #[num,t*f]
			particle_values = self.obj_func(adv_sample,metric,device)	#get object functions value

			#update best particle
			pos = particle_values < best_val
			best_pos[pos] = particles[pos]
			best_val[pos] = particle_values[pos]

			#update global best 
			global_pos = particle_values.min()<global_best_val
			global_best_pos = particles[particle_values.argmin()].clone() if global_pos else global_best_pos
			global_best_val = min(global_best_val,particle_values.min()) if global_pos else global_best_val

		sample[part] = global_best_pos
		return sample, global_best_val

	def PGD(self, metric, bounds, device, iters, step):

		ori_sample = self.test.clone().detach().requires_grad_(True)     ##xt tensor [n,t,f]
		sample = ori_sample + torch.rand(ori_sample.shape).to(device)  ##x0 tensor [n,t,f]
		ori_saliency = getAttribution(ori_sample, self.original_label, self.model, self.saliency_method)  #tensor [n,t,f]
		for  i in range(iters):
			saliency = getAttribution(sample, self.original_label, self.model, self.saliency_method)  #tensor [n,t,f]
			if metric == "MC":
				loss = torch.norm(self.cal_mass_center(saliency[0],device)-self.mass_center)
				loss.requires_grad_(True)
				direction = torch.sign(torch.autograd.grad(loss, ori_sample,retain_graph=True)[0])
				pert = step * direction
				sample = sample + pert
				sample = torch.min(torch.max(sample.flatten(),bounds[:,0]),bounds[:,1])	#clip
				sample = sample.reshape(ori_sample.shape)
			if metric == "topK":
				ele = torch.zeros_like(saliency.reshape(-1))
				ele[self.topK]=1
				loss = torch.sum(saliency.reshape(-1)*ele)
				# top = torch.argsort(saliency.reshape(-1))[-self.k_top:]
				# intersection = torch.unique(top)[torch.isin(torch.unique(top),torch.unique(self.topK))].size(0)
				# loss= intersection/self.k_top
				# loss.requires_grad_(True)
				direction = torch.sign(torch.autograd.grad(loss, ori_sample,retain_graph=True)[0])
				pert = step * direction
				sample = sample + pert
				sample = torch.min(torch.max(sample.flatten(),bounds[:,0]),bounds[:,1])	#clip
				sample = sample.reshape(ori_sample.shape)
		return sample.detach()

	def obj_func(self,particles,metric,device):
		particles = particles.reshape(particles.shape[0],self.shape[1],self.shape[2]) #[num,t,f]
		pred = self.predict(particles)  #[num,1]
		constrain = abs(pred - self.original_label.to(device))<0.0005 #[num,1], true or flase
		criterion = torch.ones(pred.shape[0]).to(device)  #[num]  obj value
		for i in range(pred.shape[0]):
			if constrain[i] == True: 
				attr = getAttribution(particles[i].reshape(1,particles.shape[1],particles.shape[2]), self.original_label, self.model, self.saliency_method)
				# attr = get_smooth_saliency(particles[i], self.shape[1], self.shape[2], device, 50, "DBA", self.model, self.saliency_method)
				attr = torch.Tensor(attr).to(device)
				if torch.any(torch.isnan(attr))==False:
					if metric == "PCC":
						attr_flatten = attr.reshape(-1)
						criterion[i] = torch.abs(pearson_cor(attr_flatten,self.saliency_flatten))
					if metric == "MC":
						center = self.cal_mass_center(attr[0],device)
						criterion[i] = -torch.norm(self.mass_center-center)
					if metric == "topK":
						top = torch.argsort(attr.reshape(-1))[-self.k_top:]
						criterion[i] = np.intersect1d(top.cpu().numpy(), self.topK.cpu().numpy()).size/self.k_top
					if metric == "DTW":
						attr_np = attr.cpu().numpy()
						saliency_np = self.saliency.cpu().numpy()
						output = 0
						for j in range(saliency_np.shape[1]):
							distance, _ = fastdtw(attr_np[:,j], saliency_np[:,j], dist = euclidean)
							output = output + distance
						criterion[i] = - output/saliency_np.shape[1]
			else:
				criterion[i] = 1 
		return criterion

def pearson_cor(x,y):
	vx = x - torch.mean(x)
	vy = y - torch.mean(y)
	rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
	return rho
