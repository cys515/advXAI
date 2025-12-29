import sys
from pathlib import Path
import numpy as np
import torch
import argparse 
import dill
from joblib import Parallel, delayed
import torch.utils.data as data_utils
from fastdtw import fastdtw
from sklearn.feature_selection import mutual_info_regression
from scipy.sparse import csr_matrix, save_npz, load_npz, eye as sp_eye
from scipy.sparse.linalg import cg
from explain import getAttribution

class SpatioTemporalCausal:
	def __init__(self, use_memmap=False, max_lag=3, n_jobs=-1,
				 causal_threshold=0.7, sparsify=False,
				 cache_dir="/storage/d05/cys/CYS/robust-timepoint/Results/Matrix/"):
		"""
		支持：缓存、无稀疏化（默认）、λ-投影。
		"""
		self.max_lag = max_lag
		self.n_jobs = n_jobs
		self.threshold = causal_threshold
		self.use_memmap = use_memmap
		self.sparsify = sparsify

		self.dependency_matrix = None   # ndarray 或 csr_matrix
		self.cache_dir = Path(cache_dir)
		self.cache_dir.mkdir(parents=True, exist_ok=True)

	# ---------- 缓存 ----------
	def _cache_paths(self, data_name: str):
		"""返回该数据集名对应的稀疏/致密两种缓存路径"""
		dense_path = self.cache_dir / f"{data_name}_Cprior_dense.npy"
		sparse_path = self.cache_dir / f"{data_name}_Cprior_sparse.npz"  # <- 用 .npz
		return dense_path, sparse_path

	def load_from_cache(self, data_name: str) -> bool:
		dense_path, sparse_path = self._cache_paths(data_name)

		# 目标是“稀疏”
		if self.sparsify:
			if sparse_path.exists():
				self.dependency_matrix = load_npz(sparse_path)
				print(f"[Cache] Loaded SPARSE prior matrix: {sparse_path}")
				return True
			if dense_path.exists():
				# 从致密派生稀疏
				dep = np.load(dense_path)
				mask = dep > self.threshold
				dep_sparse = csr_matrix(dep * mask, copy=False)
				self.dependency_matrix = dep_sparse
				# 顺手缓存为稀疏文件，之后可直接命中
				save_npz(sparse_path, dep_sparse)
				print(f"[Cache] Derived SPARSE from DENSE and saved: {sparse_path}")
				return True
			return False

		# 目标是“致密”
		else:
			if dense_path.exists():
				self.dependency_matrix = np.load(dense_path)
				print(f"[Cache] Loaded DENSE prior matrix: {dense_path}")
				return True
			if sparse_path.exists():
				# 从稀疏转致密
				dep_sparse = load_npz(sparse_path)
				dep_dense = dep_sparse.toarray()
				self.dependency_matrix = dep_dense
				# 可选：缓存致密版，后续更快
				np.save(dense_path, dep_dense)
				print(f"[Cache] Loaded SPARSE and densified, saved DENSE: {dense_path}")
				return True
			return False


	def save_to_cache(self, data_name: str):
		dense_path, sparse_path = self._cache_paths(data_name)
		if hasattr(self.dependency_matrix, "toarray"):  # 稀疏
			save_npz(sparse_path, self.dependency_matrix)
			print(f"[Cache] Saved sparse prior matrix to: {sparse_path}")
		else:
			np.save(dense_path, self.dependency_matrix)
			print(f"[Cache] Saved dense prior matrix to: {dense_path}")

	# ---------- 构建时滞依赖矩阵 ----------
	def _compute_pairwise_te(self, X_slice, i, j, tau):
		cause = X_slice[:, i]
		effect = X_slice[:, j]
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
		n_samples, n_time, n_feat = X.shape
		total_nodes = n_time * n_feat
		dep = np.zeros((total_nodes, total_nodes), dtype=np.float32)

		tasks = [(t, f, delta_t)
				 for t in range(n_time)
				 for f in range(n_feat)
				 for delta_t in range(1, self.max_lag + 1)
				 if t + delta_t < n_time]

		def _worker(X, t, f, delta_t):
			X_slice_src = X[:, t, :]
			scores = []
			for target_f in range(n_feat):
				te = self._compute_pairwise_te(X_slice_src, f, target_f, delta_t)
				scores.append(te)
			self_te = self._compute_pairwise_te(X_slice_src, f, f, delta_t)
			return (t, f, delta_t), np.asarray(scores, dtype=np.float32), self_te

		results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
			delayed(_worker)(X, t, f, delta_t) for (t, f, delta_t) in tasks
		)

		for (t, f, delta_t), scores, self_te in results:
			source_idx = t * n_feat + f
			target_t = t + delta_t
			for target_f, s in enumerate(scores):
				target_idx = target_t * n_feat + target_f
				dep[target_idx, source_idx] = s
			target_idx_self = target_t * n_feat + f
			dep[target_idx_self, source_idx] = self_te

		if self.sparsify:
			mask = dep > self.threshold
			dep = csr_matrix(dep * mask, copy=False)

		self.dependency_matrix = dep

	def fit(self, X, data_name: str = None):
		if data_name is not None and self.load_from_cache(data_name):
			return self

		assert X.ndim == 3, "X should be [N, T, F]"
		if self.use_memmap:
			X = np.memmap('/tmp/X_memmap', dtype=X.dtype, mode='w+', shape=X.shape)
		if not X.flags.writeable:
			X = np.copy(X); X.setflags(True)

		print("[Build] Building temporal dependency matrix "
			  f"({'sparse' if self.sparsify else 'dense'})...")
		self._build_time_series_matrix(X)

		if data_name is not None:
			self.save_to_cache(data_name)
		return self

	# ---------- 投影求解（λ） ----------
	def _project_dense(self, A_flat, T, F, lambda_val):
		"""
		稠密情形：解 (I + λ (I - C^T)^T (I - C^T)) z = A_flat
		"""
		C = self.dependency_matrix  # [TF, TF] ndarray
		I = np.eye(C.shape[0], dtype=C.dtype)
		Ct = C.T
		M = (I - Ct).T @ (I - Ct)
		K = I + (lambda_val * M)
		# 解线性方程
		z = np.linalg.solve(K, A_flat)
		return z.reshape(T, F)

	def _project_sparse(self, A_flat, T, F, lambda_val):
		"""
		稀疏情形：用 CG 解 (I + λ (I - C^T)^T (I - C^T)) z = A_flat
		"""
		C = self.dependency_matrix.tocsr()
		TF = C.shape[0]
		I = sp_eye(TF, format='csr', dtype=np.float32)
		Ct = C.transpose().tocsr()
		M = (I - Ct).transpose().tocsr().dot(I - Ct).tocsr()
		# K = I + λ M
		# 使用线性算子避免构成过大稀疏矩阵（也可直接 K = I + lambda*M 再 cg）
		K = I + (lambda_val * M)
		z, info = cg(K, A_flat.astype(np.float32), maxiter=200, tol=1e-5)
		if info != 0:
			print(f"[Warn] CG not fully converged (info={info}).")
		return z.reshape(T, F)

	# ---------- 接口：transform ----------
	def transform(self, attr, lambda_val=None):
		"""
		当 lambda_val 为 None：采用老的平均法（向后兼容）
		当 lambda_val 为 数值：执行投影法（论文公式）
		"""
		if isinstance(attr, torch.Tensor):
			return self._transform_tensor(attr, lambda_val)
		else:
			return self._transform_numpy(attr, lambda_val)

	def _transform_numpy(self, attr, lambda_val):
		original_shape = attr.shape  # [T, F] or [N, T, F]
		if attr.ndim == 2:
			attr = attr[np.newaxis, ...]
		N, T, F = attr.shape
		out = np.empty_like(attr)

		# 选择模式：平均 or 投影
		is_sparse = hasattr(self.dependency_matrix, "toarray")
		for n in range(N):
			A = attr[n]
			if lambda_val is None:
				# 平均法（旧版行为）
				dep = self.dependency_matrix.toarray().T if is_sparse else self.dependency_matrix.T
				enhanced = (A.reshape(1, -1) @ dep).reshape(T, F)
				out[n] = (enhanced + A) / 2.0
			else:
				A_flat = A.reshape(-1)
				if is_sparse:
					out[n] = self._project_sparse(A_flat, T, F, lambda_val)
				else:
					out[n] = self._project_dense(A_flat, T, F, lambda_val)

		return out[0] if original_shape == (T, F) else out

	def _transform_tensor(self, attr, lambda_val):
		original_shape = attr.shape
		if attr.dim() == 2:
			attr = attr.unsqueeze(0)
		N, T, F = attr.shape
		outs = []
		for n in range(N):
			A_np = attr[n].detach().cpu().numpy()
			A_proj = self._transform_numpy(A_np, lambda_val)
			outs.append(torch.tensor(A_proj, dtype=attr.dtype, device=attr.device))
		out = torch.stack(outs, dim=0)
		return out if original_shape == (N, T, F) else out[0]

	def visualize_causal_graph(self):
		pass

def main(args):
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	with open(args.data_dir + args.DataName + "120.dill", 'rb') as f:
		dataset = dill.load(f)

	Training, TrainingLabel, Testing, TestingLabel = dataset[0], dataset[1], dataset[2], dataset[3]
	t = Training.shape[1]
	f = Training.shape[2]

	test_dataRNN = data_utils.TensorDataset(torch.from_numpy(Testing), torch.from_numpy(TestingLabel))
	test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=1, shuffle=False)

	# model
	saveModelName = args.model_dir + args.model + "/" + args.DataName
	saveModelBestName = saveModelName + "_" + str(args.step) + "_BEST.pkl"
	model = torch.load(saveModelBestName, map_location=device)

	# 解析 lambdas（逗号分隔）
	lambdas = [s for s in args.lambdas.split(",") if s.strip() != ""]
	lambdas = [float(x) for x in lambdas]
	print("[Lambda] grid =", lambdas)

	# 关系矩阵（缓存优先）
	causal_masker = SpatioTemporalCausal(
		sparsify=True,  # 如需稀疏化对照改 True
		cache_dir=args.causal_dir
	)
	causal_masker.fit(Training, data_name=args.DataName)
	print("Temporal dependency matrix ready (cached or built).")

	# 为每个 λ 分别计算并保存
	out_dir = Path(args.Smooth_Saliency_dir) / args.DataName / args.model
	out_dir.mkdir(parents=True, exist_ok=True)

	for lam in lambdas:
		print(f"[Run] lambda = {lam}")
		smooth_saliency = np.zeros((25, t, f), dtype=np.float32)

		for i, (sample, label) in enumerate(test_loaderRNN):
			if i == 25:
				break
			sample = sample.to(device)
			label = label.to(device)
			attr = getAttribution(sample, label, model, args.XAI).squeeze(0).detach().cpu().numpy()  # [t, f]

			robust_attr = causal_masker.transform(attr, lambda_val=lam)
			smooth_saliency[i, :, :] = robust_attr

		# 保存：按 λ 区分文件
		lam_str = str(lam).replace('.', 'p')
		np.save(out_dir / f"{args.XAI}_lam{lam_str}.npy", smooth_saliency)
		print(f"[Save] {out_dir / f'{args.XAI}_lam{lam_str}.npy'}")



def predict(model, input):
	model.eval()
	outputs = model(input)
	return outputs

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--DataName', type=str, default="PRSA")
	parser.add_argument('--model', type=str, default="LSTM")
	parser.add_argument('--XAI', type=str, default="LIME")
	parser.add_argument('--generate', type=str, default="DBA")

	parser.add_argument('--device', type=str, default='cuda:2')
	parser.add_argument('--step', type=int, default=120)
	parser.add_argument('--N', type=int, default=100)
	
	parser.add_argument('--data_dir', type=str, default="/storage/d05/cys/CYS/robust-timepoint/Datasets/")
	parser.add_argument('--model_dir', type=str, default="/storage/d05/cys/CYS/robust-timepoint/Models/")
	parser.add_argument('--Smooth_Saliency_dir', type=str, default='/storage/d05/cys/CYS/robust-timepoint/Results/Smooth_Saliency/')
	parser.add_argument('--causal_dir', type=str, default='/storage/d05/cys/CYS/robust-timepoint/Results/Matrix/')

	parser.add_argument('--model_list', type=str, default='model_list.txt') 
	parser.add_argument('--time_log', type=str, default='timelog.txt')
	parser.add_argument('--lambdas', type=str, default='0.2,0.4,0.6,0.8')
	return  parser.parse_args()

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))  
	