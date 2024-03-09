from Cif100_labb import labels_dictionary
class Class_cifar():
	def __init__(self, classnum):
		self.classnum = classnum
		self.lab_dict = labels_dictionary
	def Get(self):
		item = self.lab_dict[self.classnum]
		return item