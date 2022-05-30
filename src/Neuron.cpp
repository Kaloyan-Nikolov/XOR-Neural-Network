#include "Neuron.h"

int Neuron::neuronsCnt = 0;

Neuron::Neuron()
{
	id = neuronsCnt++;
	bias = 0;
}

Neuron::Neuron(double newBias)
{
	id = neuronsCnt++;
	bias = newBias;
}

Neuron::Neuron(double newBias, vector<int> newIncomingEdges, vector<int> newOutgoingEdges)
{
	id = neuronsCnt++;
	bias = newBias;
	incomingEdges = newIncomingEdges;
	outgoingEdges = newOutgoingEdges;
}

int Neuron::getId()
{
	return id;
}

double Neuron::getBias()
{
	return bias;
}

double Neuron::getValue()
{
	return value;
}

double Neuron::getFoundError()
{
	return foundError;
}

double Neuron::getFutureChangeBias()
{
	return futureChangeBias;
}

vector<int> Neuron::getIncomingEdges()
{
	return incomingEdges;
}

vector<int> Neuron::getOutgoingEdges()
{
	return outgoingEdges;
}

void Neuron::setBias(double newBias)
{
	bias = newBias;
}

void Neuron::setValue(double newValue)
{
	value = newValue;
}

void Neuron::setFoundError(double newFoundError)
{
	foundError = newFoundError;
}

void Neuron::setFutureChangeBias(double change)
{
	futureChangeBias = change;
}

void Neuron::addIncomingEdge(int from)
{
	incomingEdges.push_back(from);
}

void Neuron::addOutgoingEdge(int to)
{
	outgoingEdges.push_back(to);
}
