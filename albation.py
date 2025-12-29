import sys
import argparse
import numpy as np
import torch
import dill
import torch.utils.data as data_utils
from explain import getAttribution

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from fastdtw import fastdtw
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse import lil_matrix, csr_matrix
import sys
import numpy as np
from fastdtw import fastdtw


class SpatioTemporalCausal:
    def __init__(self, use_memmap=False, max_lag=3, n_jobs=-1,
                 causal_threshold=0.7, sparsify=False):
        """
        Dense (no-sparsification) by default.
        Set sparsify=True if you ever want to restore threshold/pruning later.

        :param max_lag: 最大时间滞后步数
        :param n_jobs: 并行计算核数
        :param causal_threshold: （仅在 sparsify=True 时生效）阈值
        :param sparsify: 是否进行稀疏化（本次消融：False）
        """
        self.max_lag = max_lag
        self.n_jobs = n_jobs
        self.threshold = causal_threshold
        self.use_memmap = use_memmap
        self.sparsify = sparsify

        self.dependency_matrix = None   # 致密或稀疏的时滞依赖矩阵
        self.time_window = None

    def _compute_pairwise_te(self, X_slice, i, j, tau):
        """
        计算单步（给定 t）的 i->j, lag=tau 的依赖分数（使用 MI）
        X_slice: 形状 [N, F]，对应固定时间 t 的样本批，与你原始用法一致
        """
        # 对齐时间序列（正滞后）
        # 这里按你原代码只用正向滞后 delta_t>0
        cause = X_slice[:, i]
        effect = X_slice[:, j]

        # FastDTW 对齐（逐样本拼接到一条序列的近似；与你原代码保持风格一致）
        # 说明：这里对每个时间点使用同一对齐路径的近似做法，保留你原始写法思路
        _, path = fastdtw(cause, effect, radius=5)
        path = np.array(path).T
        aligned_cause = cause[path[0]]
        aligned_effect = effect[path[1]]

        mi = mutual_info_regression(
            aligned_cause.reshape(-1, 1),
            aligned_effect,
            n_neighbors=5
        )
        return float(mi[0])

    def _build_time_series_matrix(self, X):
        """
        构建【致密】时滞依赖矩阵（无稀疏化版本）。
        形状: total_nodes x total_nodes, 其中节点索引 = t * F + f
        """
        n_samples, n_time, n_feat = X.shape
        total_nodes = n_time * n_feat

        # 用 dense 矩阵承载（无阈值过滤，无剪枝）
        dep = np.zeros((total_nodes, total_nodes), dtype=np.float32)

        # 任务列表：所有 (t, f, delta_t)
        tasks = [(t, f, delta_t)
                 for t in range(n_time)
                 for f in range(n_feat)
                 for delta_t in range(1, self.max_lag + 1)
                 if t + delta_t < n_time]

        # 并行计算：对每个 (t,f,delta_t) 计算该源节点到所有 target_f 的分数
        def _worker(X, t, f, delta_t):
            # X[:, t, :] -> [N, F]
            X_slice_src = X[:, t, :]
            X_slice_tgt = X[:, t + delta_t, :]
            # 为与原写法一致，这里统一用 X_slice_src 传入（i,j 的索引来自不同时间）
            scores = []
            for target_f in range(n_feat):
                te = self._compute_pairwise_te(X_slice_src, f, target_f, delta_t)
                scores.append(te)
            # 自连接（同一特征的时间依赖）：可复用一次（保持原设计）
            self_te = self._compute_pairwise_te(X_slice_src, f, f, delta_t)
            return (t, f, delta_t), np.asarray(scores, dtype=np.float32), self_te

        results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(_worker)(X, t, f, delta_t) for (t, f, delta_t) in tasks
        )

        # 回填到矩阵（所有分数都写入，不做阈值筛选）
        for (t, f, delta_t), scores, self_te in results:
            source_idx = t * n_feat + f
            target_t = t + delta_t
            # 写入所有目标特征
            for target_f, s in enumerate(scores):
                target_idx = target_t * n_feat + target_f
                dep[target_idx, source_idx] = s
            # 自身跨时依赖（可覆盖上面的同索引项，不影响）
            target_idx_self = target_t * n_feat + f
            dep[target_idx_self, source_idx] = self_te

        # 如果用户强制要求稀疏化（本次消融不需要）
        if self.sparsify:
            from scipy.sparse import csr_matrix
            mask = dep > self.threshold
            dep = csr_matrix(dep * mask, copy=False)

        self.dependency_matrix = dep  # 可能是 ndarray 或 csr_matrix

    def fit(self, X):
        """估计时滞依赖矩阵（无稀疏化）"""
        assert X.ndim == 3, "X should be [N, T, F]"

        if self.use_memmap:
            X = np.memmap('/tmp/X_memmap', dtype=X.dtype, mode='w+', shape=X.shape)

        if not X.flags.writeable:
            X = np.copy(X)
            X.setflags(write=True)

        self._build_time_series_matrix(X)
        return self

    def transform(self, attr):
        """同时支持 NumPy / PyTorch"""
        if isinstance(attr, torch.Tensor):
            return self._transform_tensor(attr)
        else:
            return self._transform_numpy(attr)

    def _transform_numpy(self, attr):
        original_shape = attr.shape  # [T, F] or [N, T, F]
        if attr.ndim == 2:
            attr = attr[np.newaxis, ...]
        # 形状对齐为 [N, T*F]
        N, T, F = attr.shape
        node_attr = attr.reshape(N, -1)

        if hasattr(self.dependency_matrix, "toarray"):  # 稀疏情形（仅当 sparsify=True）
            dep = self.dependency_matrix.toarray().T
        else:
            dep = self.dependency_matrix.T  # ndarray

        enhanced = node_attr @ dep
        enhanced = enhanced.reshape(N, T, F)
        out = (enhanced + attr) / 2.0
        return out[0] if original_shape == (T, F) else out

    def _transform_tensor(self, attr):
        original_shape = attr.shape  # [T, F] or [N, T, F]
        if attr.dim() == 2:
            attr = attr.unsqueeze(0)
        N, T, F = attr.shape
        node_attr = attr.reshape(N, -1)

        if hasattr(self.dependency_matrix, "toarray"):  # 稀疏情形（仅当 sparsify=True）
            dep_np = self.dependency_matrix.toarray().T
        else:
            dep_np = self.dependency_matrix.T

        dep_tensor = torch.tensor(dep_np, dtype=attr.dtype, device=attr.device)
        enhanced = node_attr @ dep_tensor
        enhanced = enhanced.view(N, T, F)
        out = (enhanced + attr) / 2.0
        return out if original_shape == (N, T, F) else out[0]

    # 可选的可视化函数保留（不影响本次消融）
    def visualize_causal_graph(self):
        pass  # 本次不涉及

	def __init__(self, use_memmap=False,  max_lag=3, n_jobs=-1, causal_threshold=0.7):
		"""
		:param max_lag: 最大时间滞后步数
		:param n_jobs: 并行计算核数
		:param causal_threshold: 因果强度阈值
		"""
		self.max_lag = max_lag
		self.n_jobs = n_jobs
		self.threshold = causal_threshold
		self.causal_matrix = None
		self.time_window = None
		self.use_memmap = use_memmap

	def _compute_pairwise_te(self, X, i, j, tau):
		"""计算时滞转移熵"""
		n_samples = X.shape[0]
		valid_lag = abs(tau)
		
		# 对齐时间序列
		if tau > 0:
			cause = X[:-tau, i]
			effect = X[tau:, j]
		else:
			cause = X[-tau:, i]
			effect = X[:tau, j]
		
		# 使用FastDTW对齐
		_, path = fastdtw(cause, effect, radius=5)
		path = np.array(path).T
		aligned_cause = cause[path[0]]
		aligned_effect = effect[path[1]]
		
		# 计算转移熵
		mi = mutual_info_regression(
		aligned_cause.reshape(-1, 1), 
		aligned_effect,
		n_neighbors=5  # 关键参数调整
	)
		return mi[0]

	def _build_time_series_graph(self, X):
		"""构建时空因果图（带并行安全处理）"""
		n_samples, n_time, n_feat = X.shape
		total_nodes = n_time * n_feat
		self.causal_matrix = lil_matrix((total_nodes, total_nodes))
		
		# 创建任务参数元组（避免传递整个数组）
		tasks = [(t, f, delta_t) 
				for t in range(n_time) 
				for f in range(n_feat)
				for delta_t in range(1, self.max_lag+1)]
		
		# 修改并行任务调度
		results = Parallel(n_jobs=self.n_jobs, 
						 max_nbytes=None)(  # 禁用自动内存映射
			delayed(self._compute_temporal_causality)(X, t, f, delta_t)
			for t, f, delta_t in tasks
		)
		
		# 填充稀疏矩阵
		for (source_idx, target_idx), strength in results:
			if strength > self.threshold:
				self.causal_matrix[target_idx, source_idx] = strength
		self.causal_matrix = self.causal_matrix.tocsr()

	def _compute_temporal_causality(self, X, t, f, delta_t):
		"""计算单个时间特征点的因果影响"""
		X = np.copy(X)
		X.setflags(write=True)

		n_samples, n_time, n_feat = X.shape
		source_idx = t * n_feat + f
		
		# 仅考虑未来时间窗口
		if t + delta_t >= n_time:
			return (None, None), 0.0
		
		strengths = []
		for target_f in range(n_feat):
			target_t = t + delta_t
			target_idx = target_t * n_feat + target_f
			
			# 计算跨时间因果
			te = self._compute_pairwise_te(X[:, t, :], f, target_f, delta_t)
			strengths.append(te)
			
			# 计算时间自相关
			if f == target_f:
				te_self = self._compute_pairwise_te(X[:, t, :], f, f, delta_t)
				self.causal_matrix[target_idx, source_idx] = te_self
		
		max_strength = max(strengths)
		target_idx = (t + delta_t) * n_feat + np.argmax(strengths)
		return (source_idx, target_idx), max_strength

	def fit(self, X):
		"""训练时空因果图"""
		assert X.ndim == 3
		
		if self.use_memmap:
			# 使用内存映射避免内存复制
			X = np.memmap('/tmp/X_memmap', dtype=X.dtype, 
						mode='w+', shape=X.shape)
		
		# 确保数据可写
		if not X.flags.writeable:
			X = np.copy(X)
			X.setflags(write=True)
			
		self._build_time_series_graph(X)
		return self

	def _prune_isolated_nodes(self):
		"""修剪孤立节点"""
		node_degrees = self.causal_matrix.sum(axis=1).A1
		isolated = node_degrees < np.percentile(node_degrees, 10)
		self.causal_matrix[isolated, :] = 0
		self.causal_matrix[:, isolated] = 0

	def transform(self, attr):
		"""同时支持NumPy和PyTorch输入"""
		if isinstance(attr, torch.Tensor):
			return self._transform_tensor(attr)
		else:
			return self._transform_numpy(attr)

	def _transform_numpy(self, attr):
		# 原有处理NumPy的逻辑
		original_shape = attr.shape
		if attr.ndim == 2:
			attr = attr[np.newaxis, ...]
		node_attr = attr.reshape(-1, self.causal_matrix.shape[0])
		enhanced = node_attr @ self.causal_matrix.T.A  # 转换为密集矩阵
		return (enhanced.reshape(original_shape) + attr) / 2

	def _transform_tensor(self, attr):
		# 新增处理PyTorch张量的逻辑
		original_shape = attr.shape
		if attr.dim() == 2:
			attr = attr.unsqueeze(0)
		
		# 将稀疏矩阵转换为PyTorch张量
		causal_tensor = torch.tensor(self.causal_matrix.A, 
								dtype=attr.dtype,
								device=attr.device)
		
		node_attr = attr.reshape(attr.size(0), -1)
		enhanced = node_attr @ causal_tensor.T
		return (enhanced.view(original_shape) + attr) / 2

	def visualize_causal_graph(self):
		"""可视化关键因果路径（示例实现）"""
		import networkx as nx
		G = nx.DiGraph(self.causal_matrix.toarray())
		
		# 提取重要路径
		pr = nx.pagerank(G)
		top_nodes = sorted(pr.items(), key=lambda x: -x[1])[:10]
		
		# 构建子图
		subgraph = G.subgraph([n[0] for n in top_nodes])
		nx.draw(subgraph, with_labels=True)
def main(args):
	
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	with open(args.data_dir+args.DataName+"120.dill", 'rb') as f:
		dataset = dill.load(f)
		
	Training,TrainingLabel,Testing,TestingLabel= dataset[0],dataset[1],dataset[2],dataset[3]

	t = Training.shape[1]
	f = Training.shape[2]

	test_dataRNN = data_utils.TensorDataset(torch.from_numpy(Testing),torch.from_numpy(TestingLabel))
	test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=1, shuffle=False)

	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	saveModelName=args.model_dir+args.model+"/"+args.DataName
	saveModelBestName =saveModelName +"_"+str(args.step)+"_BEST.pkl"
	model = torch.load(saveModelBestName,map_location=device)   
	smooth_saliency = np.zeros([25,t,f])
	 
	causal_masker = SpatioTemporalCausal()
	print("Building causal graph...")
	causal_masker.fit(Training)  # [N, T, F]
	
	for i, (sample, label) in enumerate(test_loaderRNN):
		if i==25:
			break
		print("sample:", i)
		sample = sample.to(device)
		label = label.to(device)
		attr = getAttribution(sample, label, model, args.XAI).squeeze(0).detach().cpu().numpy()  # [t, f]
		
		robust_attr = causal_masker.transform(attr)
		
		smooth_saliency[i,:,:] = robust_attr

	np.save(args.Smooth_Saliency_dir + args.DataName+"/"+args.model+"/"+args.XAI, smooth_saliency)



def predict(model, input):
	model.eval()
	outputs = model(input)
	return outputs

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--DataName', type=str, default="PRSA")
	parser.add_argument('--model', type=str, default="LSTM")
	parser.add_argument('--XAI', type=str, default="SM")
	parser.add_argument('--generate', type=str, default="DBA")

	parser.add_argument('--device', type=str, default='cuda:2')
	parser.add_argument('--step', type=int, default=120)
	parser.add_argument('--N', type=int, default=100)
	
	parser.add_argument('--data_dir', type=str, default="/storage/d05/cys/CYS/robust-timepoint/Datasets/")
	parser.add_argument('--model_dir', type=str, default="/storage/d05/cys/CYS/robust-timepoint/Models/")
	parser.add_argument('--Smooth_Saliency_dir', type=str, default='/storage/d05/cys/CYS/robust-timepoint/Results/Smooth_Saliency/')

	parser.add_argument('--model_list', type=str, default='model_list.txt') 
	parser.add_argument('--time_log', type=str, default='timelog.txt')
	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))  
	