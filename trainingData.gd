##########################################  class TrainingData  ################################################
extends Node

var m_trainingDataFile


func isEof():
	return m_trainingDataFile.eof_reached()

func getTopology(topology):
	var line
	var label
	var ss
	
	line = m_trainingDataFile.get_csv_line(" ")
	if isEof() or not line[0] == "topology:":
		print("fatal input error (file error)")
		assert false
	line.remove(0)
	return line
	
	#for i in range(line.size() - 1):
	#	var n = float(line[i + 1])
	#	topology.append(n)

func TrainingData(filename):
	m_trainingDataFile = File.new()
	m_trainingDataFile.open(filename, m_trainingDataFile.READ)

func getNextInputs(inputVals):
	
	var line = m_trainingDataFile.get_csv_line(" ") #m_trainingDataFile.get_line()
	
	if line[0] == "in:":
		line.remove(0)
		line.resize(line.size() - 1) # TODO please fix file and remove this line
		return line
	else:
		print("file error")

func getTargetOutputs(targetOutputVals):
	
	var line = m_trainingDataFile.get_csv_line(" ")
	if line[0] == "out:":
		line.remove(0)
		return line
	else:
		print("file error")