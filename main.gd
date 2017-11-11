##########################################  class Main  ################################################
extends Node

var Net = preload("res://Net.tscn")
var TrainingData = preload("res://TrainingData.tscn")

var trainData
var myNet
var topology
var inputVals
var targetVals
var resultVals
var trainingPass

func _ready():
	
	inputVals = Array()
	targetVals = Array()
	resultVals = Array()
	topology = Array()
	
	trainData = TrainingData.instance()
	trainData.TrainingData("res://trainingData.txt")
	topology = trainData.getTopology(topology)
	
	myNet = Net.instance()
	myNet._ready()
	myNet.Net(topology)
	
	trainingPass = 0
	
	while not trainData.isEof():
		trainingPass += 1
		print("Pass ", trainingPass)
		
		# Get new input data and feed it forward:
		inputVals = trainData.getNextInputs(inputVals)
		if not inputVals.size() == int(topology[0]):
			print("error: next input != topology[0]")
			break
		showVectorVals("Inputs:", inputVals)
		myNet.feedForward(inputVals) 
		
		# Collect the net's actual output results:
		resultVals = myNet.getResults(resultVals)
		showVectorVals("Outputs:", resultVals)
		
		# Train the net what the outputs should have been:
		targetVals = trainData.getTargetOutputs(targetVals)
		showVectorVals("Targets:", targetVals)
		assert(targetVals.size() == int(topology[topology.size() - 1]))
		
		myNet.backProp(targetVals)
		
		# Report how well the training is working, average over recent samples:
		print("Net recent average error: ", myNet.getRecentAverageError())
	print("done")

func showVectorVals(label, v):
	print (label, " ", v)


