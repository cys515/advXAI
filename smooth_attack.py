import numpy as np
import time
from torch.autograd import Variable
import torch
import scipy
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from explain import getAttribution
from smooth import *
import multiprocessing
from scipy.optimize import approx_fprime, differential_evolution
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class Attack(object):
	
	def __init__(self, args, test, label, model, saliency, saliency_method, device, causal_masker, lam, k_top=10):
		self.test = test	#tensor [1,t,f]
		self.shape = test.shape
		self.original_label = label.reshape(label.shape[0],1).to(device)	#tensor [num,1]
		self.model = model
		self.saliency = torch.from_numpy(saliency).to(device)	#tensor [t,f]
		self.test_flatten = self.test.reshape(-1)		#tensor [t*f]
		self.saliency_flatten = self.saliency.reshape(-1)	#tensor [t*f]
		self.saliency_method = saliency_method
		self.causal_masker = causal_masker
		self.k_top = k_top
		self.lam = lam
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
			attack_method: [PSO | DE]
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
			sample, criterion = self.GA(metric, bounds, part, device, iters)

		if attack_method == "PGD":
			sample = self.PGD(metric, bounds, device, iters, step = 0.005)
		sample = sample.reshape(self.shape)
		label = self.predict(sample)
		attr = getAttribution(sample, label, self.model, self.saliency_method)
		attr = self.causal_masker.transform(attr,lambda_val=self.lam)
		return sample.detach().cpu().numpy(), attr.detach().cpu().numpy()

	def GA(self, metric, bounds, part, device, iterations=100, pop_size=40, 
	   mutation_rate=0.1, elite_ratio=0.1):
	
		d = bounds.shape[0]
		adv_sample = self.test_flatten.repeat(pop_size, 1)  # [pop_size, t*f]
		sample = self.test_flatten.clone()
		
		# initialize
		population = (torch.rand(pop_size, d).to(device) * (bounds[:,1]-bounds[:,0]) + bounds[:,0])  # [pop_size, d]
		
		# evaluate
		adv_sample[:, part] = population
		fitness = self.obj_func(adv_sample, metric, device)
		
		best_idx = fitness.argmin()
		global_best = population[best_idx].clone()
		global_fitness = fitness[best_idx]

		for gen in range(iterations):
			# choose
			probs = torch.softmax(1/(fitness + 1e-9), dim=0)  
			parents_idx = torch.multinomial(probs, pop_size, replacement=True)
			parents = population[parents_idx]

			# cross
			crossover_point = torch.randint(1, d, (pop_size//2,))
			mask = torch.zeros_like(parents, dtype=torch.bool)
			for i, cp in enumerate(crossover_point):
				mask[2*i, :cp] = True
				mask[2*i+1, cp:] = True
			offspring = torch.where(mask, parents, parents.roll(1, dims=0))

			mutation_mask = torch.rand_like(offspring) < mutation_rate
			noise = torch.randn_like(offspring) * 0.1 * (bounds[:,1]-bounds[:,0])
			population = torch.clamp(offspring + noise * mutation_mask, bounds[:,0], bounds[:,1])

			elite_num = int(pop_size * elite_ratio)
			elite_idx = fitness.topk(elite_num, largest=False).indices
			population[:elite_num] = population[elite_idx]

			adv_sample[:, part] = population
			fitness = self.obj_func(adv_sample, metric, device)

			current_best_idx = fitness.argmin()
			if fitness[current_best_idx] < global_fitness:
				global_best = population[current_best_idx].clone()
				global_fitness = fitness[current_best_idx]

		sample[part] = global_best
		return sample, global_fitness


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
		ori_saliency = self.causal_masker.transform(ori_saliency, lambda_val=self.lam)
		for  i in range(iters):
			saliency = getAttribution(sample, self.original_label, self.model, self.saliency_method)  #tensor [n,t,f]
			saliency = self.causal_masker.transform(saliency, lambda_val=self.lam)
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
				attr = self.causal_masker.transform(attr, lambda_val=self.lam).reshape(particles.shape[1],particles.shape[2])
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
						attr_np = attr.cpu().detach().numpy()
						saliency_np = self.saliency.cpu().numpy()
						output = 0
						for j in range(saliency_np.shape[1]):
							distance, _ = fastdtw(attr_np[:,j].astype(float).reshape(-1, 1), saliency_np[:,j].astype(float).reshape(-1, 1), dist = euclidean)
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