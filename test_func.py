import numpy as np

class test_func():
	def __init__(self, data_name='ASAP'):
		self.dataset = data_name
		if self.dataset == 'ASAP':
			self.cluster_map = {0: 'Service', 1: 'Food', 2: 'Service', 3: 'None', 4: 'Food', 5: 'Ambience',
								6: 'Price', 7: 'Ambience', 8: 'Location', 9: 'None', 10: 'None', 11: 'Food',
								12: 'Food', 13: 'None', 14: 'Food', 15: 'Food', 16: 'Ambience', 17: 'Ambience'}
			self.order = ['Location', 'Service', 'Price', 'Ambience', 'Food']
		elif self.dataset == 'TA':
			self.cluster_map = {0: 'stuff', 1: 'value', 2: 'room', 3: 'clean', 4: 'location', 5: 'service',
								6: 'service', 7: 'room', 8: 'location', 9: 'None', 10: 'service', 11: 'room',
								12: 'location', 13: 'service', 14: 'value', 15: 'business', 16: 'None', 17: 'location',
								18: 'service', 19: 'None'}
			self.order = ['value', 'room', 'location', 'clean', 'stuff', 'service', 'business']
		elif self.dataset == 'GS':
			self.cluster_map = {0: 'Bullying policy', 1: 'Academics', 2: 'None', 3: 'Character', 4: 'None',
								5: 'Programs', 6: 'LD support', 7: 'None', 8: 'None', 9: 'None', 10: 'None',
								11: 'None', 12: 'None', 13: 'Teachers', 14: 'None', 15: 'Leadership', 16: 'None',
								17: 'None', 18: 'None', 19: 'None'}
			self.order = ['Bullying policy', 'Character', 'LD Support', 'Leadership', 'Teachers']


	# aspect identification
	def evaluate_asp_identification(self, asp_imp, asp_info, th):
		def aspImp_2_labels(asp_imp, th):
			dic = {}
			if self.dataset == 'ASAP':
				dic = {'Location': 0., 'Service': 0., 'Price': 0., 'Ambience': 0., 'Food': 0., 'None': 0.}
			elif self.dataset == 'TA':
				dic = {'value': 0, 'room': 0, 'location': 0, 'clean': 0, 'stuff': 0, 'service': 0, 'business':0, 'None': 0}
			elif self.dataset == 'GS':
				dic = {'Academics': 0, 'Bullying policy': 0, 'Character': 0, 'Programs': 0, 'LD support': 0, 'Teachers': 0, 'Leadership': 0, 'None': 0}

			for i in range(len(asp_imp)):
				if asp_imp[i] > dic[self.cluster_map[i]]:
					dic[self.cluster_map[i]] = asp_imp[i]
			dic.pop('None')
			r = {k: v for k, v in dic.items() if v >= th}
			labels = list(r.keys())
			return labels

		def aspInfo_2_golds(asp_info):
			g = []
			for i in range(len(asp_info)):
				if asp_info[i] != -2:
					g.append(self.order[i])
			return g

		golds = aspInfo_2_golds(asp_info)
		labels = aspImp_2_labels(asp_imp, th)

		return golds, labels


	def evaluate_ACSA(self, asp_imp, asp_senti, asp_info, best_th):
		def aspInfo_2_golds(asp_info):
			golds = []
			for i in range(len(asp_info)):
				if asp_info[i] != -2:
					golds.append((self.order[i], asp_info[i]))

			return golds

		def aspSenti_2_labels(asp_imp, asp_senti, gold_keys):
			dic = {}
			if self.dataset == 'ASAP':
				dic = {'Location': 0., 'Service': 0., 'Price': 0., 'Ambience': 0., 'Food': 0., 'None': 0.}
			elif self.dataset == 'TA':
				dic = {'value': 0, 'room': 0, 'location': 0, 'clean': 0, 'stuff': 0, 'service': 0, 'business': 0, 'None': 0}
			elif self.dataset == 'GS':
				dic = {'Academics': 0, 'Bullying policy': 0, 'Character': 0, 'Programs': 0, 'LD support': 0, 'Teachers': 0, 'Leadership': 0, 'None': 0}

			for i in range(len(asp_imp)):
				if asp_imp[i] > dic[self.cluster_map[i]]:
					dic[self.cluster_map[i]] = asp_imp[i]
			dic.pop('None')
			r = {k: v for k, v in dic.items() if v >= best_th}
			labels = list(r.keys())
			AC_dic = {}
			if self.dataset == 'ASAP':
				AC_dic = {'Location': [0., 0., 0.], 'Service': [0., 0., 0.], 'Price': [0., 0., 0.], 'Ambience': [0., 0., 0.],
						  'Food': [0., 0., 0.], 'None': [0., 0., 0.]}
			elif self.dataset == 'TA':
				AC_dic = {'value': [0., 0., 0.], 'room': [0., 0., 0.], 'location': [0., 0., 0.], 'clean': [0., 0., 0.],
						  'stuff': [0., 0., 0.], 'service': [0., 0., 0.], 'business': [0., 0., 0.]}
			elif self.dataset == 'GS':
				AC_dic = {'Academics': [0., 0., 0.], 'Bullying policy': [0., 0., 0.], 'Character': [0., 0., 0.],
						  'Programs': [0., 0., 0.], 'LD support': [0., 0., 0.], 'Teachers': [0., 0., 0.],
						  'Leadership': [0., 0., 0.], 'None': [0., 0., 0.]}

			for i in range(len(asp_imp)):
				if self.cluster_map[i] in labels:
					item = [j * asp_imp[i] for j in asp_senti[i]]
					re = AC_dic[self.cluster_map[i]]
					AC_dic[self.cluster_map[i]] = [re[j] + item[j] for j in range(3)]

			res_1 = []
			for i in AC_dic:
				if AC_dic[i] != [0., 0., 0.]:
					v = np.argmax(AC_dic[i])
					res_1.append((i, v - 1))  # [0, 1, 2] -> [-1, 0, 1]
			res_2 = []
			for i in AC_dic:
				if i in gold_keys:
					if AC_dic[i] != [0., 0., 0.]:
						v = np.argmax(AC_dic[i])
						res_2.append((i, v - 1))
					else:
						res_2.append((i, -2))
			return res_1, res_2

		gold = aspInfo_2_golds(asp_info)
		gold_keys = [i[0] for i in gold]
		label_AC, label_SC = aspSenti_2_labels(asp_imp, asp_senti, gold_keys)

		return gold, label_AC, label_SC