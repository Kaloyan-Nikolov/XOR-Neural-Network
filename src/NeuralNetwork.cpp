#include "NeuralNetwork.h"

mt19937 NeuralNetwork::initializeRandomGenerator()
{
	std::random_device rd;
	std::mt19937 g(rd());
	return g;
}

double NeuralNetwork::getRandomNumberDouble(double lowerbound, double upperbound)
{
	std::uniform_real_distribution<double> distribution(lowerbound, upperbound);
	return distribution(gen);
}

void NeuralNetwork::initializeSeparateNeurons(int numInputLayer, int numHiddenLayer, int numOutputLayer)
{
	double currBias;
	for (int i = 0; i < numInputLayer; i++)
	{
		currBias = 0;
		Neuron next(currBias);
		inputLayer.push_back(next);
	}
	for (int i = 0; i < numHiddenLayer; i++)
	{
		currBias = 0;
		Neuron next(currBias);
		hiddenLayer.push_back(next);
	}
	for (int i = 0; i < numOutputLayer; i++)
	{
		currBias = -1;
		Neuron next(currBias);
		outputLayer.push_back(next);
	}

	inputLayerSize = numInputLayer;
	hiddenLayerSize = numHiddenLayer;
	outputLayerSize = numOutputLayer;
}

void NeuralNetwork::initializeNeuronLinks()
{
	double randomWeight;
	for (int i = 0; i < inputLayerSize; i++)
	{
		for (int j = 0; j < hiddenLayerSize; j++)
		{
			randomWeight = getRandomNumberDouble(-0.05, 0.05);
			edges.insert(std::pair<int, Edge>(Edge::getFirstVacantId(), Edge(inputLayer[i].getId(), hiddenLayer[j].getId(), randomWeight)));
		}
	}
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < outputLayerSize; j++)
		{
			randomWeight = getRandomNumberDouble(-0.05, 0.05);
			edges.insert(std::pair<int, Edge>(Edge::getFirstVacantId(), Edge(hiddenLayer[i].getId(), outputLayer[j].getId(), randomWeight)));
		}
	}
}

void NeuralNetwork::initializeEdges()
{
	for (int i = 0; i < inputLayerSize; i++)
	{
		// there are no incoming edges
		for (int j = 0; j < hiddenLayerSize; j++)
		{
			inputLayer[i].addOutgoingEdge(getEdgeIdByFromAndTo(inputLayer[i].getId(), hiddenLayer[j].getId()));
		}
	}
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		for (int j = 0; j < inputLayerSize; j++)
		{
			hiddenLayer[i].addIncomingEdge(getEdgeIdByFromAndTo(inputLayer[j].getId(), hiddenLayer[i].getId()));
		}
		for (int j = 0; j < outputLayerSize; j++)
		{
			hiddenLayer[i].addOutgoingEdge(getEdgeIdByFromAndTo(hiddenLayer[i].getId(), outputLayer[j].getId()));
		}
	}
	for (int i = 0; i < outputLayerSize; i++)
	{
		for (int j = 0; j < hiddenLayerSize; j++)
		{
			outputLayer[i].addIncomingEdge(getEdgeIdByFromAndTo(hiddenLayer[j].getId(), outputLayer[i].getId()));
		}
		// there are no outgoing edges
	}
}

void NeuralNetwork::initializeChanges()
{
	for (auto it = edges.begin(); it != edges.end(); it++)
	{
		it->second.setFutureChange(0);
	}
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		hiddenLayer[i].setFutureChangeBias(0);
	}
	for (int i = 0; i < outputLayerSize; i++)
	{
		outputLayer[i].setFutureChangeBias(0);
	}
}

int NeuralNetwork::getEdgeIdByFromAndTo(const int f, const int t)
{
	for (auto it = edges.begin(); it != edges.end(); it++)
	{
		if (it->second.getFrom() == f && it->second.getTo() == t)
		{
			return it->first;
		}
	}
}

double NeuralNetwork::sigmoid(double x)
{
	return 1/(1+std::exp(-x));
}

double NeuralNetwork::inputFunction(Neuron & neuron, const int neuronLayer)
{
	double sum = 0;
	int currEdgeId;
	double currEdgeWeight;
	double currEdgeFromValue;
	for (int i = 0; i < neuron.getIncomingEdges().size(); i++)
	{
		currEdgeId = neuron.getIncomingEdges()[i];
		auto currEdgePtr = edges.find(currEdgeId);
		currEdgeWeight = currEdgePtr->second.getWeight();
		currEdgeFromValue = getNeuronValueByIdAndLayer(currEdgePtr->second.getFrom(), neuronLayer - 1);
		sum += currEdgeWeight * currEdgeFromValue;
	}
	return sum;
}

double NeuralNetwork::getNeuronValueByIdAndLayer(const int id, const int layer)
{
	if (layer == 1)
	{
		for (int i = 0; i < inputLayerSize; i++)
		{
			if (inputLayer[i].getId() == id)
			{
				return inputLayer[i].getValue();
			}
		}
	}
	else if (layer == 2)
	{
		for (int i = 0; i < hiddenLayerSize; i++)
		{
			if (hiddenLayer[i].getId() == id)
			{
				return hiddenLayer[i].getValue();
			}
		}
	}
	else if(layer == 3)
	{
		for (int i = 0; i < outputLayerSize; i++)
		{
			if (outputLayer[i].getId() == id)
			{
				return outputLayer[i].getValue();
			}
		}
	}
}

double NeuralNetwork::getNeuronValueById(const int id)
{
	for (int i = 0; i < inputLayerSize; i++)
	{
		if (inputLayer[i].getId() == id)
		{
			return inputLayer[i].getValue();
		}
	}
	for (int i = 0; i < hiddenLayerSize; i++)
	{
		if (hiddenLayer[i].getId() == id)
		{
			return hiddenLayer[i].getValue();
		}
	}
	for (int i = 0; i < outputLayerSize; i++)
	{
		if (outputLayer[i].getId() == id)
		{
			return outputLayer[i].getValue();
		}
	}
}

void NeuralNetwork::setValueOfNeuron(Neuron & neuron, const int neuronLayer)
{
	double inputSum = inputFunction(neuron, neuronLayer);
	double value = sigmoid(inputSum + neuron.getBias());
	neuron.setValue(value);	
	if(neuronLayer == 3)
		bestRes[cntBestRes++] = value;
}

double NeuralNetwork::findSumWeightAndError(Neuron & neuron, const int neuronLayer)
{
	double sum = 0;
	int currEdgeId;
	double currEdgeWeight;
	double currEdgeToFoundError;
	for (int i = 0; i < neuron.getOutgoingEdges().size(); i++)
	{
		currEdgeId = neuron.getOutgoingEdges()[i];
		auto currEdgePtr = edges.find(currEdgeId);
		currEdgeWeight = currEdgePtr->second.getWeight();
		currEdgeToFoundError = getNeuronFoundErrorByIdAndLayer(currEdgePtr->second.getTo(), neuronLayer + 1);
		sum += currEdgeWeight * currEdgeToFoundError;
	}
	return sum;
}

double NeuralNetwork::getNeuronFoundErrorByIdAndLayer(const int id, const int layer)
{
	if (layer == 1)
	{
		for (int i = 0; i < inputLayerSize; i++)
		{
			if (inputLayer[i].getId() == id)
			{
				return inputLayer[i].getFoundError();
			}
		}
	}
	else if (layer == 2)
	{
		for (int i = 0; i < hiddenLayerSize; i++)
		{
			if (hiddenLayer[i].getId() == id)
			{
				return hiddenLayer[i].getFoundError();
			}
		}
	}
	else if (layer == 3)
	{
		for (int i = 0; i < outputLayerSize; i++)
		{
			if (outputLayer[i].getId() == id)
			{
				return outputLayer[i].getFoundError();
			}
		}
	}
}

NeuralNetwork::NeuralNetwork(const vector<pair<double, double>>& givenInputValues, const vector<double>& givenExpectedOutput,
	int numInputLayer, int numHiddenLayer, int numOutputLayer, double givenLearningRate)
{
	inputValues = givenInputValues;
	expectedOutput = givenExpectedOutput;
	learningRate = givenLearningRate;
	bestRes.resize(inputValues.size());
	gen = initializeRandomGenerator();
	initializeSeparateNeurons(numInputLayer, numHiddenLayer, numOutputLayer);
	initializeNeuronLinks();
	initializeEdges();
	initializeChanges();
}

void NeuralNetwork::epoch()
{
	double currError = 0;
	error = 0;
	cntBestRes = 0;
	for (int i = 0; i < inputValues.size(); i++)
	{
		inputLayer[0].setValue(inputValues[i].first);
		inputLayer[1].setValue(inputValues[i].second);

		// find yj -> find value
		for (int j = 0; j < hiddenLayerSize; j++)
		{
			setValueOfNeuron(hiddenLayer[j], 2);
		}
		double myCurrError;
		for (int j = 0; j < outputLayerSize; j++)
		{
			setValueOfNeuron(outputLayer[j], 3);
			myCurrError = outputLayer[0].getValue() - expectedOutput[i];
			error += pow(myCurrError, 2);
		}

		// find delta
		double currDeltaOutput;
		for (int i = 0; i < outputLayerSize; i++)
		{
			currDeltaOutput = myCurrError * outputLayer[i].getValue() * (1 - outputLayer[i].getValue());
			outputLayer[i].setFoundError(currDeltaOutput);

			// w3, w4
			int currEdgeId;
			double change;
			double currValueFromHidden;
			double wjPlus;
			for (int j = 0; j < outputLayer[i].getIncomingEdges().size(); j++)
			{
				currEdgeId = outputLayer[i].getIncomingEdges()[j];
				auto currEdgePtr = edges.find(currEdgeId);
				currValueFromHidden = getNeuronValueById(currEdgePtr->second.getFrom());
				change = currDeltaOutput * currValueFromHidden;
				currEdgePtr->second.setFutureChange(currEdgePtr->second.getFutureChange() + change);

				//wjPlus = currEdgePtr->second.getWeight() - learningRate * change;
			}
		}

		double changeW;
		double sumWkhDeltaK;
		double prod;
		double delta;
		for (int i = 0; i < hiddenLayerSize; i++)
		{
			sumWkhDeltaK = findSumWeightAndError(hiddenLayer[i], 2);
			prod = hiddenLayer[i].getValue() * (1 - hiddenLayer[i].getValue());

			delta = sumWkhDeltaK * prod;

			// w01, w11, w02, w12
			double wjPlus;
			int currEdgeId;
			double currInput;
			for (int j = 0; j < hiddenLayer[i].getIncomingEdges().size(); j++)
			{
				currEdgeId = hiddenLayer[i].getIncomingEdges()[j];
				auto currEdgePtr = edges.find(currEdgeId);
				currInput = getNeuronValueById(currEdgePtr->second.getFrom());
				changeW = delta * currInput;

				currEdgePtr->second.setFutureChange(currEdgePtr->second.getFutureChange() + changeW);

				//wjPlus = currEdgePtr->second.getWeight() - learningRate * changeW;
			}
		}
	}

	// update the weight of edges
	double stepSize;
	for (auto it = edges.begin(); it != edges.end(); it++)
	{
		stepSize = learningRate * it->second.getFutureChange();
		it->second.updateWeight(stepSize);
		it->second.setFutureChange(0);
	}
}

void NeuralNetwork::train()
{
	double minError = 0.00000001;
	error = 1;
	for (int i = 0; i < 1000001 && error > minError; i++)
	{
		epoch();
		if (i % 100000 == 0)
		{ 
			printErrorAndBestResult();
		}
	}
}

void NeuralNetwork::printErrorAndBestResult()
{
	cout << setw(15) << left << error << "| ";
	for (int i = 0; i < 4; i++)
	{
		cout << setw(15) << left << bestRes[i] << " ";
	}
	cout << "\n";
}
