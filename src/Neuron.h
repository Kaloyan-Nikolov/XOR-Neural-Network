#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
	int id;
	double bias;
	double value;
	double foundError;
	double futureChangeBias;
	vector<int> incomingEdges; // connect neurons from previous layer with the current neuron
	vector<int> outgoingEdges; // connect the current neuron with neurons from the next layer

	static int neuronsCnt;

public:
	Neuron();
	Neuron(double newBias);
	Neuron(double newBias, vector<int> newIncomingEdges, vector<int> newOutgoingEdges);

	int getId();
	double getBias();
	double getValue();
	double getFoundError();
	double getFutureChangeBias();
	vector<int> getIncomingEdges();
	vector<int> getOutgoingEdges();

	void setBias(double newBias);
	void setValue(double newValue);
	void setFoundError(double newFoundError);
	void setFutureChangeBias(double change);
	void addIncomingEdge(int from);
	void addOutgoingEdge(int to);
};