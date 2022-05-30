#pragma once
#include "Neuron.h"

class Edge
{
	int id;
	int from;
	int to;
	double weight;
	double futureChange;

	static int edgesCnt;

public:
	Edge();
	Edge(int f, int t, double w);

	int getId();
	int getFrom();
	int getTo();
	double getWeight();
	double getFutureChange();

	void setFutureChange(const double change);
	void updateWeight(const double change);

	static int getFirstVacantId();
};