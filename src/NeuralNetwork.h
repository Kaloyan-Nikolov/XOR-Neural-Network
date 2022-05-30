#pragma once
#include <random>
#include <cmath>
#include <map>
#include <iomanip>
#include "Edge.h"

class NeuralNetwork
{
	vector<Neuron> inputLayer;
	vector<Neuron> hiddenLayer;
	vector<Neuron> outputLayer;
	map<int, Edge> edges;

	int inputLayerSize;
	int hiddenLayerSize;
	int outputLayerSize;
	mt19937 gen;

	vector<pair<double, double>> inputValues;
	vector<double> expectedOutput;

	double learningRate;
	vector<double> bestRes;
	int cntBestRes = 0;

	double error;

	mt19937 initializeRandomGenerator();
	double getRandomNumberDouble(double lowerbound, double upperbound);

	void initializeSeparateNeurons(int numInputLayer, int numHiddenLayer, int numOutputLayer);
	void initializeNeuronLinks();
	void initializeEdges();
	void initializeChanges();
	int getEdgeIdByFromAndTo(const int f, const int t);

	double sigmoid(double x);
	double inputFunction(Neuron& neuron, const int neuronLayer);
	double getNeuronValueByIdAndLayer(const int id, const int layer);
	double getNeuronValueById(const int id);
	void setValueOfNeuron(Neuron& neuro, const int neuronLayer);

	double findSumWeightAndError(Neuron& neuron, const int neuronLayer);
	double getNeuronFoundErrorByIdAndLayer(const int id, const int layer);

public:
	NeuralNetwork(const vector<pair<double, double>>& givenInputValues, const vector<double>& givenExpectedOutput, 
		int numInputLayer = 2, int numHiddenLayer = 2, int numOutputLayer = 1, double givenLearningRate = 0.5);

	void epoch();
	void train();
	void printErrorAndBestResult();
};