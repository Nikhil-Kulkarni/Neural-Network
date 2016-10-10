import random
import math

class Neuron:
	def __init__(self, numInputs):
		self.bias = random.uniform(-0.2, 0.2)
		self.weights = []
		self.errors = []
		for i in range(0, numInputs):
			self.weights.append(random.uniform(-0.2,0.2))

	def process(self, inputs):
		self.lastInputs = inputs
		sum = 0
		for i in range(0, len(inputs)):
			sum += inputs[i] * self.weights[i]
		sum += self.bias
		self.lastOuput = self.sigmoid(sum)
		return self.lastOuput

	def sigmoid(self, num):
		return 1 / (1 + pow(math.e, -1 * num))

class Layer:
	def __init__(self, numNeurons, numInputs):
		self.neurons = []
		for i in range(0, numNeurons):
			self.neurons.append(Neuron(numInputs))

	def process(self, inputs):
		result = []
		for i in range(0, len(self.neurons)):
			result.append(self.neurons[i].process(inputs))
		return result

class Network:
	def __init__(self):
		self.layers = []
		self.errorThreshold = 0.00001
		self.trainingIterations = 500000
		self.learningRate = 0.3

	def process(self, inputs):
		outLay = []
		for lay in self.layers:
			outLay = lay.process(inputs)
			inputs = outLay
		return outLay

	def addLayer(self, numNeurons, numInputs):
		if numInputs == None:
			numInputs = len(self.layers[len(self.layers) - 1].neurons)
		self.layers.append(Layer(numNeurons, numInputs))

	# def meansqerr(self, error):


	def train(self, examples):
		outputLayer = self.layers[len(self.layers) - 1]
		for it in range(0, self.trainingIterations):
			for i in range(0, len(examples)):
				input = examples[i][0]
				target = examples[i][1]
				output = self.process(input)

				for n in range(0, len(outputLayer.neurons)):
					neuron = outputLayer.neurons[n]
					neuron.error = target[n] - output[n]

					neuron.errors.append(neuron.error)
					neuron.delta = neuron.lastOuput * (1 - neuron.lastOuput) * neuron.error

				for l in range(len(self.layers) - 2, -1, -1):
					for j in range(0, len(self.layers[l].neurons)):
						currNeuron = self.layers[l].neurons[j]
						# Backprop the error and delta
						sum = 0
						for k in range(0, len(self.layers[l+1].neurons)):
							sum += self.layers[l+1].neurons[k].weights[j] * self.layers[l+1].neurons[k].error
						currNeuron.error = sum
						currNeuron.delta = currNeuron.lastOuput * (1 - currNeuron.lastOuput) * currNeuron.error
						# Adjust weights and bias
						for k in range(0, len(self.layers[l+1].neurons)):
							nNeuron = self.layers[l+1].neurons[k]
							for w in range(0, len(nNeuron.weights)):
								nNeuron.weights[w] += self.learningRate * nNeuron.lastInputs[w] * nNeuron.delta
							nNeuron.bias = self.learningRate * nNeuron.delta






		