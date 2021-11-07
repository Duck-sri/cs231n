import numpy as np
from typing import List

def sigmoid(z): return 1/(1+np.exp(-1*z))

class MyFC:
	def __init__(self,layers:List[int],activation=sigmoid,learning_rate=0.01,bias:bool=False):
		# model params
		self.L:int = len(layers)
		self.layers = layers
		self.learning_rate:float = learning_rate

		# related func
		self.activation_func = activation

		# weight init
		self.ws = [None]
		self.a = [None] + [np.zeros((i,1)) for i in self.layers[1:]]
		self.z = [None] + [np.zeros((i,1)) for i in self.layers[1:]]
		if bias: 
			self.bs = [None] + [np.random.randn(i,1) for i in layers[1:]]
		for i in range(1,self.L):
			w = np.random.randn(layers[i],layers[i-1])
			self.ws.append(w)
			if bias: self.bs.append(np.random.randn(layers[i],1))


	def activation_func_diff(self,z):
		return self.activation_func(z)*(1-self.activation_func(z))

	def zero_grad(self):
		self.dw = [np.zeros_like(x) for x in self.ws]
		self.delta = [np.zeros((i,1)) for i in self.layers]
		if self.bs: self.db = [np.zeros_like(x) for x in self.bs]

	def init_grad(self):
		self.zero_grad()
			
	def forward(self,x:np.ndarray,disp=False):
		assert x.shape[0]==self.ws[1].shape[-1],f"Wrong input shape, Expected : {self.ws[1].shape[-1]}, got: {x.shape[0]}"
		self.a[0] = x
		for l in range(1,self.L):
			tmp = f"{self.ws[l].shape}*{x.shape} + {self.bs[l].shape}"
			self.z[l] = self.ws[l]@self.a[l-1] + self.bs[l]
			self.a[l] = self.activation_func(self.z[l])
			tmp += f" = {self.a[l].shape}"
			if disp: print(tmp)
		return self.a[-1]

	def backprop(self,loss):
		self.delta[-1] = loss
		m = loss.shape[-1]
		for l in range(1,self.L+1,-1):
			#TODO do with autograd support
			
			self.dw[l] = 1/m * (self.delta[l]@self.a[l-1])
			self.db[l] = 1/m * np.sum(self.delta[l],axis=1,keepdims=True)
			self.delta[l-1] = (self.delta[l]@self.a[l])*(self.activation_func_diff(self.z[l-1]))

			self.ws[l] -= self.learning_rate*(self.dw[l])
			self.bs[l] -= self.learning_rate*(self.db[l])
		print("BackProp done...")

	def trainStep(self,X,y,epochs:int=5):
		for i in range(epochs):
			yHat = self.forward(X)
			J = (y-yHat).reshape(-1,1)
			self.zero_grad()
			self.backprop(J)

	def __call__(self,*args,**kwargs):
		return self.forward(*args,**kwargs)

	def __repr__(self):
		ans = f"No.of layers : {self.L} "
		if self.bs:
			for i,(w,b) in enumerate(zip(self.ws,self.bs)):
				ans+= "\n" + (f"W[{i}]:{w.shape} B[{i}]:{b.shape}")
		else:
			for i,w in enumerate(self.ws):
				ans+= "\n" + (f"W[{i}]:{w.shape}")
		return ans

layers = list(np.random.randint(1,10,size=(10,)))
model = MyFC(layers,bias=True)
X = np.random.randn(layers[0],10)
y = np.random.randn(layers[-1],10)
print("__ Layers __")
print(layers)
print("__ Forward Prop __")
y_hat = model(X,disp=True)
print("__ yHat Shape __")
print(y_hat.shape)

print("BackProp")
model.trainStep(X,y)