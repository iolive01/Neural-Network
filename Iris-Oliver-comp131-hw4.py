# ANN
# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
# I collaborated with Leah Stern, Cathy Cowell, Supriya Sanjay, Ki Ki Chan

from __future__ import print_function

import numpy
import math

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Used to create colors, reference 
# https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
#
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def sigmoid(potential):
	if (abs(potential) > 700):
		potential = 700

	return 1 / (1 + math.exp(-potential))

# one layer class represents two layers and their connections: a top 
# layer of nodes and their connections to the bottom layer of nodes.  
class Layer:

	def __init__(self, numBottom, numTop):
		self.inputs = numpy.zeros([1, numBottom])

		# random weights between -1 and 1
		self.weights = 2 * numpy.random.rand(numBottom, numTop) - 1 
		self.potentials = numpy.zeros([1, numTop])
		self.outputs = numpy.zeros([1, numTop])
		self.topErrors = numpy.zeros([1, numTop])
		self.bottomErrors = numpy.zeros([1, numBottom])

	def printLayer(self):
		print("inputs")
		print(self.inputs)
		print("weights:")
		print(self.weights)
		print("potentials:")
		print(self.potentials)
		print("outputs:")
		print(self.outputs)

	# takes in a list of inputs, len must correspond to numBottom
	# copies these inputs into self.inputs
	def loadInputs(self, inputList):
		self.inputs = numpy.squeeze(self.inputs)
		for i in range(len(inputList)):
			self.inputs[i] = inputList[i]

	# fills the potentials array, calculates the potentials
	def calcPotentials(self):
		self.potentials = numpy.squeeze(self.potentials)
		for i in range(self.potentials.shape[0]): 	   # for each potential value
			product = self.weights[:,i] * self.inputs  # relevant weights * inputs, 1x4
			sigma = numpy.sum(product)		 		   # potential value
			self.potentials[i] = sigma			   # store in relevant spot

	# calculates outputs and fills the outputs array - front prop
	def calcOutputs(self):
		self.outputs = numpy.squeeze(self.outputs)
		self.calcPotentials()
		for i in range(self.outputs.shape[0]):
			self.outputs[i] = sigmoid(self.potentials[i])

	# loads in the errors that pertain to the top half, len corr to numtop
	def loadTopErrors(self, errorList):
		self.topErrors = numpy.squeeze(self.topErrors)
		for i in range(len(errorList)):
			self.topErrors[i] = errorList[i]

	# calculates the bottom errors and puts in an array of size bottom
	def calcBottomErrors(self):
		self.bottomErrors = numpy.squeeze(self.bottomErrors)
		for i in range(self.bottomErrors.shape[0]): # rows
			product = self.weights[i,:] * self.outputs
			sigma = numpy.sum(product)
			self.bottomErrors[i] = sigma

	# does the backpropagation algorithm
	def backPropagate(self):
		self.calcBottomErrors()
		learningFactor = 1
		deltas = numpy.squeeze(numpy.multiply.outer(self.inputs, self.topErrors))
		deltas *= learningFactor
		self.weights += deltas
		self.weights = numpy.squeeze(self.weights)


def normalizeData(flowerData):
	normalized = flowerData / flowerData.max(axis=0) * 6
	return normalized

def inputData(flowerData, types):
	file = open('irisData.txt', 'r')
	if (file.mode == 'r'):
		lineNum = 0
		data = file.readlines()
		for flower in data:
			irisArr = flower.split(',')
			
			# load the sepal and petal data
			for i in range(4):
				flowerData[lineNum, i] = irisArr[i]

			# load the classification
			irisArr[4] = irisArr[4].replace('\n', '')
			types.append(irisArr[4])
			lineNum += 1

def calcErrors(target, actual):
	errors = numpy.zeros([1, 3])
	errors = numpy.squeeze(errors)
	for i in range(len(actual)):
		errors[i] = actual[i] * (1 - actual[i]) * (target[i] - actual[i])
	return errors

# maps the given name to a 1x3 matrix of binary values
# iris setosa 	  = [1, 0, 0]
# iris versicolor = [0, 1, 0]
# iris virginica  = [0, 0, 1]
def nameToBinary(string):
	if (string == 'Iris-setosa'):
		return [1, 0, 0]
	elif (string == 'Iris-versicolor'):
		return [0, 1, 0]
	else:
		return [0, 0, 1]

def binaryApproxToName(array):
	maxPos = numpy.argmax(array)
	if (maxPos == 0):
		return "Iris-setosa"
	elif (maxPos == 1):
		return "Iris-versicolor"
	else:
		return "Iris-virginica"

def printAnalysis(targetName, targetBinary, actualBinary):
	actualName = binaryApproxToName(actualBinary)
	if (targetName == actualName):
		print(color.GREEN, "Correct! ", color.END, "This is an ", actualName, ".", sep="")
		print("Network outputted: ", actualBinary.tolist(), \
				" Actual binary: ", targetBinary, sep='')
		return True
	else:
		print(color.RED, "Not quite! ", color.END, "This is an ", targetName, " not an ", actualName, ".", sep="")
		print("Network outputted: ", actualBinary.tolist(), \
				" Actual binary: ", targetBinary, sep='')
		return False

def runNetwork():
	petalMeasures = numpy.zeros([150, 4])
	classifications = []
	inputData(petalMeasures, classifications)
	petalMeasures = normalizeData(petalMeasures)

	inputToHidden = Layer(4, 5)
	hiddentoOutput = Layer(5, 3)
	
	# Training phase
	print(color.PURPLE, color.BOLD, "Time to Train!", color.END)
	numCorrect = 0
	for q in range(10):
		for i in range(len(classifications)):
			# load what expected should be
			target = nameToBinary(classifications[i])

			inputToHidden.loadInputs(petalMeasures[i,:])
			inputToHidden.calcOutputs()

			hiddentoOutput.loadInputs(inputToHidden.outputs)
			hiddentoOutput.calcOutputs()

			outcome = printAnalysis(classifications[i], target, hiddentoOutput.outputs)
			if (outcome == True):
				numCorrect += 1

			errors = calcErrors(target, hiddentoOutput.outputs)

			hiddentoOutput.loadTopErrors(errors)
			hiddentoOutput.backPropagate()

			inputToHidden.loadTopErrors(hiddentoOutput.bottomErrors)
			inputToHidden.backPropagate()

	print(color.PURPLE, color.BOLD, "Training over. Of 1500 classifications, ", numCorrect, " were correct.", color.END, sep="")

	# Classification phase
	# for some reason it calculates everything as the same as the Iris-virginicas which were trained
	# last. I tried lowering the learning factor, but this didn't change anything. 

	print("The network is now trained. Please enter details about your iris below")
	sepalLength = input('Enter the sepal length in cm: ')
	sepalWidth = input('Enter the sepal width in cm: ')
	petalLength = input('Enter the petal length in cm: ')
	petalWidth = input('Enter the petal width in cm: ')

	userInput = [sepalLength, sepalWidth, petalLength, petalWidth]

	inputToHidden.loadInputs(userInput)
	inputToHidden.calcOutputs()

	hiddentoOutput.loadInputs(inputToHidden.outputs)
	hiddentoOutput.calcOutputs()

	classification = binaryApproxToName(hiddentoOutput.outputs)
	print(color.BLUE, color.BOLD, "I think this is an ", classification, ".", sep="")
	print("Here's the full analysis:")
	print(hiddentoOutput.outputs)
	print(nameToBinary(classification), color.END, sep="")


runNetwork()



