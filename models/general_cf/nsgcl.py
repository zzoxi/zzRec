import torch as t
import numpy as np
import scipy.sparse as sp
import random
from torch import nn
from scipy.sparse import csr_matrix
from config.configurator import configs
from models.aug_utils import EmbedPerturb
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class NsGCL(LightGCN):
	# 初始化
	def __init__(self, data_handler):
		super(NsGCL, self).__init__(data_handler)

		self.adj = data_handler.torch_adj # lightgcn的adj
		self.mat_a = data_handler.mat_a # 用于denoise
		self.new_adj = self.gen_adj()

		self.layer_num = configs['model']['layer_num'] # 层数
		self.denoise_num = configs['model']['denoise_num']
		self.vgae_num = configs['model']['vgae_num']


		self.alpha = configs['model']['alpha'] # cl_loss权重
		self.beta = configs['model']['beta'] # cl_loss2权重
		self.reg_weight = configs['model']['reg_weight'] # reg_loss
		# tau
		self.temperature = configs['model']['temperature'] # tau
		self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
		self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
		self.latent_dim = self.embedding_size

		# 需要初始化MLP的W参数矩阵和b
		self.eps_weight = nn.Parameter(t.empty(self.latent_dim, self.latent_dim).normal_(0, 1), requires_grad=True)
		self.eps_bias = nn.Parameter(t.zeros(self.latent_dim), requires_grad=True)

		self.batch_size = configs['train']['batch_size']

		# 这是什么
		self.is_training = True
		self.final_embeds = None

	def _normalize_adj(self, mat):
		"""Laplacian normalization for mat in coo_matrix

		Args:
			mat (scipy.sparse.coo_matrix): the un-normalized adjacent matrix

		Returns:
			scipy.sparse.coo_matrix: normalized adjacent matrix
		"""
		# Add epsilon to avoid divide by zero
		degree = np.array(mat.sum(axis=-1)) + 1e-10
		d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
		d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
		return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

	# 生成一个新的邻接矩阵
	def gen_adj(self):

		# degrees是log后的度向量 mat_a 是邻接矩阵
		degrees = np.array(self.mat_a.sum(axis=1))
		degrees = np.where(degrees == 0, 0, np.log2(degrees))


		# analytic
		u_d = np.array(degrees[:self.user_num])
		i_d = np.array(degrees[self.user_num:])

		max_u_d = np.max(u_d)
		max_i_d = np.max(i_d)

		min_u_d = np.min(u_d)
		min_i_d = np.min(i_d)

		max_x = max_u_d #+ max_i_d
		min_x = min_u_d # + min_i_d

		coo_mat = self.mat_a.copy().tocoo()
		# print(type(coo_mat))
		# print
		for i, (r, c, v) in enumerate(zip(coo_mat.row, coo_mat.col, coo_mat.data)):
			if v != 0:
				# random_number = random.random()
				value = degrees[r]  + degrees[c]  # 重新计算值 主要是不能为0否则不能反向传播
				weight = (value - min_x ) /  (max_x - min_x)
				# if random_number < weight:
				coo_mat.data[i] = weight

		coo_mat = self._normalize_adj(coo_mat)
		# print(type(coo_mat))

		# count = 0
		# 遍历非零元素，根据行和列重新计算并赋值

		


		# 生成新的adj make torch tensor
		idxs = t.from_numpy(np.vstack([coo_mat.row, coo_mat.col]).astype(np.int64))
		vals = t.from_numpy(coo_mat.data.astype(np.float32))
		shape = t.Size(coo_mat.shape)
		new_adj =  t.sparse.FloatTensor(idxs, vals, shape).to(configs['device'])

		return new_adj


	# 简单lightGCN finised
	def base_forward(self):
		# 这里是测试用吗？
		if not self.is_training and self.final_embeds is not None:
			return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
		embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
		embeds_list = [embeds]
		for i in range(self.layer_num):
			embeds = t.spmm(self.new_adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)
		self.final_embeds = embeds
		return embeds[:self.user_num], embeds[self.user_num:]


	# 有两种策略 两个节点度数越高越丢弃、一个高一个低更容易丢弃

	def denoise_forward(self, user_embeds, item_embeds):
		embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
		embeds_list = [embeds]
		# 在下面使用lightgcn在新的稀疏矩阵A上进行消息聚合即可
		for i in range(self.denoise_num):
			embeds = t.spmm(self.new_adj, embeds_list[-1])
			embeds_list.append(embeds)
		embeds = sum(embeds_list)
		return embeds[:self.user_num], embeds[self.user_num:]

	# VGAE
	def vgae_forward(self, user_embeds, item_embeds):
		# 按行拼接
		embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
		embeds_list = [embeds] # 这里可不加第一层
		for i in range(self.vgae_num):
			embeds = t.spmm(self.adj, embeds_list[-1])
			embeds_list.append(embeds)

		embeds = sum(embeds_list) # / len(embeds_list)

		#print("embeds:"+str(embeds.shape))

		# 平均池化 有问题
		mean = embeds # / len(embeds_list)

		# MLP计算方差
		logstd = t.matmul(mean, self.eps_weight) + self.eps_bias
		std = t.exp(logstd)



		# 重参数
		noise = t.randn_like(std)
		noised_emb = mean + 0.01 * std * noise
		return noised_emb[:self.user_num], noised_emb[self.user_num:]

		# return user_embeds, item_embeds


	def kl_loss(self, mean, std):
		regu_loss = -0.5 * (1 + 2*std - mean**2 - std.exp()**2)
		kl_loss = t.mean(t.sum(regu_loss, 1, keepdim=True)) / self.batch_size
		return kl_loss


	def _pick_embeds(self, user_embeds, item_embeds, batch_data):
		ancs, poss, negs = batch_data
		anc_embeds = user_embeds[ancs]
		pos_embeds = item_embeds[poss]
		neg_embeds = item_embeds[negs]
		return anc_embeds, pos_embeds, neg_embeds



	# 最终的损失函数 # 我来搞个去噪层ssl试试
	def cal_loss(self, batch_data):
		# 这里有什么用
		self.is_training = True

		u = self.user_embeds
		v = self.item_embeds

		user_embeds1, item_embeds1 = self.vgae_forward(u, v)
		user_embeds2, item_embeds2 = self.vgae_forward(u, v)
		user_embeds3, item_embeds3 = self.base_forward()
		user_embeds4, item_embeds4 = self.denoise_forward(u, v)

		anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
		anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
		anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)
		anc_embeds4, pos_embeds4, neg_embeds4 = self._pick_embeds(user_embeds4, item_embeds4, batch_data)


		# bpr
		bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]

		# 计算infonce 损失
		cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.temperature) + \
		cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.temperature)
		cl_loss /= anc_embeds1.shape[0]
		cl_loss *= self.alpha

		# 1 4比更好 base_forward改成new_adj更好  晚上tu一下
		cl_loss2 = cal_infonce_loss(anc_embeds1, anc_embeds4, user_embeds4, self.temperature) + \
		cal_infonce_loss(pos_embeds1, pos_embeds4, item_embeds4, self.temperature)
		cl_loss2 /= anc_embeds1.shape[0]
		cl_loss2 *= self.beta

		# 计算reg 损失
		reg_loss = self.reg_weight * reg_params(self)



		loss = bpr_loss + reg_loss + cl_loss   + cl_loss2
		losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'cl_loss2': cl_loss2}
		return loss, losses

	def full_predict(self, batch_data):
		user_embeds, item_embeds = self.base_forward()
		# 标志位
		self.is_training = False
		pck_users, train_mask = batch_data
		pck_users = pck_users.long()
		pck_user_embeds = user_embeds[pck_users]
		full_preds = pck_user_embeds @ item_embeds.T
		full_preds = self._mask_predict(full_preds, train_mask)
		return full_preds