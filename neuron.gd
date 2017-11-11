##########################################  class Neuron  ################################################
extends Node

var Connection = {
	"weight": 1,
	"deltaWeight": 1
}

var m_outputWeights = Array()
var eta = 0.15    # [0.0..1.0] overall net training rate
var alpha = 0.5   # [0.0..n] multiplier of last weight change (momentum)
var m_myIndex
var m_gradient
var m_outputVal

func Neuron(numOutputs, myIndex):
	for c in range(numOutputs):
		m_outputWeights.append(Connection)
		m_outputWeights.back().weight = randomWeight()
	
	m_myIndex = myIndex

func updateInputWeights(prevLayer):
	# The weights to be updated are in the Connection container
    # in the neurons in the preceding layer
	
	for n in range(prevLayer.size()):
		var oldDeltaWeight = prevLayer[n].m_outputWeights[m_myIndex].deltaWeight
		
		var newDeltaWeight = eta * prevLayer[n].getOutputVal() * m_gradient + alpha * oldDeltaWeight
		# eta: Individual input, magnified by the gradient and train rate
		# alpha: momentum = a fraction of the previous delta weight
		
		prevLayer[n].m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight
		prevLayer[n].m_outputWeights[m_myIndex].weight += newDeltaWeight
	#print(m_outputWeights)
	return prevLayer

func sumDOW(nextLayer):
	var sum = 0.0
	
	#Sum our contributions of the errors at the nodes we feed.
	
	for n in range(nextLayer.size() - 1):
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient
	
	return sum

func calcHiddenGradients(nextLayer):
	var dow = sumDOW(nextLayer)
	m_gradient = dow * transferFunctionDerivative(m_outputVal)

func calcOutputGradients(targetVal):
	var delta = float(targetVal) - m_outputVal
	m_gradient = delta * transferFunctionDerivative(m_outputVal)

func transferFunction(x):
	# tanh - output range [-1.0..1.0]
	return tanh(x)

func transferFunctionDerivative(x):
	# tanh derivative
	return 1.0 - x * x # real derivative is 1 - tanhx * tanhx

func feedForward(prevLayer):
	var sum = 0.0
	
	# Sum the previous layer's outputs (which are our inputs)
	# Include the bias node from the previous layer.
	
	for n in range(prevLayer.size()):
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight
	m_outputVal = transferFunction(sum)

func setOutputVal(val):
	m_outputVal = float(val)

func getOutputVal():
	return m_outputVal

func randomWeight():
	return randf()
