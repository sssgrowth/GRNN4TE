import os
import sys
sys.path.append('..')
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertAdam
from collections import OrderedDict
import re
from action.load_embeddings import *
from utils.util import *

# MAX_SEQ=512
class Base_Model(torch.nn.Module):

	def __init__(self,args):
		super(Base_Model,self).__init__()
		self.args=args
		self.process=Process(args)
		self.util=Util()
		print('model: ',self._get_name())

		if self.args.use_bert:
			self.load_bert()
			self.load_word_embedding()
		if self.args.use_word_embeding:
			self.load_word_embedding()

	def load_word_embedding(self):
		print('load word embeddings')
		load=load_embeddings(self.args)
		self.word_embed,_,self.vocab_ix,self.char_ix=load.load_pretrained_word_embedding()
		self.word_embed_dim=self.word_embed.shape[-1]
		self.ix_vocab={i:w for w,i in self.vocab_ix.items()}
		ix_vocab=sorted(self.ix_vocab.items())
		# print(ix_vocab)
		self.vocab=[w for i,w in ix_vocab]
		self.build_vocab_ix=load.load_build_vocab()
		self.char_types=len(self.char_ix)
		self.build_size=len(self.build_vocab_ix)

	def muti_head_attention(self,q,k,v,nb_head,size_per_head,mask=None):
		
		def format_seq(x,linear,f=None):
			s=x.size()
			x=linear(x)
			t=torch.split(x,size_per_head, dim=-1) 
			s=[]
			for v in t:
				s.append(v.unsqueeze(1))
			v=torch.cat(s,dim=1)
			return v

		q=format_seq(q,self.q_linear)
		k=format_seq(k,self.k_linear)
		v=format_seq(v,self.v_linear)

		sq=q.size()
		_q=q
		_k=k
		_v=v
		_q=_q.mean(2,keepdim=True)
		factor=np.sqrt(float(size_per_head))
		a=torch.matmul(_q,torch.transpose(_k,-1,-2))/factor
		alpha=torch.nn.functional.softmax(a,-1)
		o=alpha.matmul(_v)
		o1=torch.transpose(o,1,2)
		o2=o1.contiguous().view([sq[0],1,nb_head*size_per_head])
		return o2,o2,alpha

	def init_lstm(self,input_lstm):
		"""
		Initialize lstm
		"""
		for ind in range(0, input_lstm.num_layers):
			weight = eval('input_lstm.weight_ih_l' + str(ind))
			bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
			torch.nn.init.uniform_(weight, -bias, bias)
			weight = eval('input_lstm.weight_hh_l' + str(ind))
			bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
			torch.nn.init.uniform_(weight, -bias, bias)
			
		if input_lstm.bidirectional:
			for ind in range(0, input_lstm.num_layers):
				weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
				bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
				torch.nn.init.uniform_(weight, -bias, bias)
				weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
				bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
				torch.nn.init.uniform_(weight, -bias, bias)

		if input_lstm.bias:
			for ind in range(0, input_lstm.num_layers):
				weight = eval('input_lstm.bias_ih_l' + str(ind))
				weight.data.zero_()
				weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
				weight = eval('input_lstm.bias_hh_l' + str(ind))
				weight.data.zero_()
				weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
			if input_lstm.bidirectional:
				for ind in range(0, input_lstm.num_layers):
					weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
					weight.data.zero_()
					weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
					weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
					weight.data.zero_()
					weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

	def init_embedding(self,input_embedding):
		"""
		Initialize embedding
		"""
		bias = np.sqrt(3.0 / input_embedding.size(1))
		torch.nn.init.uniform_(input_embedding, -bias, bias)

	def init_linear(self,input_linear):
		"""
		Initialize linear transformation
		"""
		bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
		torch.nn.init.uniform_(input_linear.weight, -bias, bias)
		if input_linear.bias is not None:
			input_linear.bias.data.zero_()

	def init_weight(self,w):
		torch.nn.init.xavier_uniform_(w)
