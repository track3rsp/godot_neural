##########################################  class Net  ################################################
extends Node

var Neuron = preload("res://Neuron.tscn")
var m_layers
var m_recentAverageError = 0
var m_recentAverageSmoothingFactor = 100.0
var m_error

func _ready():
	m_layers = Array()

func getRecentAverageError():
	return m_recentAverageError

func getResults(resultVals):
	resultVals.clear()
	
	for n in range(m_layers.back().size() - 1):
		resultVals.append(m_layers.back()[n].getOutputVal())
	return resultVals

func backProp(targetVals):
	var outputLayer = m_layers.back()
	m_error = 0.0
	
	# Calculate overall net error (RMS of output neuron errors)
	
	for n in range(outputLayer.size() - 1):
		var delta = float(targetVals[n]) - m_layers.back()[n].getOutputVal()
		m_error += delta * delta
	
	if not outputLayer.size() == 1: # for that to be true there should be no output layers
		m_error /= outputLayer.size() -1 # get average error squared
	else:
		print("division error")
	if not m_error < 0: # imposible cuz m_error has been squared already
		m_error = sqrt(m_error) # RMS
	else:
		print("sqrt error")
	
	# Implement a recent average measurement
	
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0)
	
	# calculate output layer gradients
	for n in range(outputLayer.size() - 1):
		m_layers.back()[n].calcOutputGradients(targetVals[n])
	
	# calculate hidden layer gradients
	for layerNum in range(m_layers.size() - 2, 0, -1):
		var hiddenLayer = m_layers[layerNum]
		var nextLayer = m_layers[layerNum + 1]
		
		for n in range(hiddenLayer.size()):
			hiddenLayer[n].calcHiddenGradients(nextLayer)
	
	# for all layers from outputs to first hidden layer 
	# update connection weights
	for layerNum in range(m_layers.size() - 1, 0, -1):
		var layer = m_layers[layerNum]
		var prevLayer = m_layers[layerNum -1]
		
		for n in range(layer.size() - 1):
			m_layers[layerNum -1] = m_layers[layerNum][n].updateInputWeights(prevLayer)

func feedForward(inputVals):
	assert(inputVals.size() == m_layers[0].size() - 1)
	
	# Assign the input values into the input neurons
	for i in range(inputVals.size()):
		m_layers[0][i].setOutputVal(inputVals[i])
	
	# Forward propagate
	for layerNum in range(1, m_layers.size()):
		var prevLayer = m_layers[layerNum - 1]
		
		for n in range(m_layers[layerNum].size() -1):
			m_layers[layerNum][n].feedForward(prevLayer)


func Net(topology):
	var numLayers = topology.size();
	for layerNum in range(numLayers):
		m_layers.append(Layer())
		var numOutputs
		if layerNum == topology.size() - 1:
			numOutputs = 0
		else:
			numOutputs = int(topology[layerNum + 1])

		# We have a new layer, now fill it with neurons, and
		# add a bias neuron in each layer.
		for neuronNum in range(int(topology[layerNum]) + 1):
			var neuron
			neuron = Neuron.instance()
			neuron.Neuron(numOutputs, neuronNum)
			m_layers.back().append(neuron)
			print ("Made a Neuron!")
		
		# Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
		m_layers.back().back().setOutputVal(1.0)

func Layer():
	var matrix = []
	#matrix.append([])
	return matrix
