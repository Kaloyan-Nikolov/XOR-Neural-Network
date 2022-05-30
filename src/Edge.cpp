#include "Edge.h"

int Edge::edgesCnt = 0;

Edge::Edge()
{
	id = edgesCnt++;
	from = 0;
	to = 0;
	weight = 0;
}

Edge::Edge(int f, int t, double w)
{
	id = edgesCnt++;
	from = f;
	to = t;
	weight = w;
}

int Edge::getId()
{
	return id;
}

int Edge::getFrom()
{
	return from;
}

int Edge::getTo()
{
	return to;
}

double Edge::getWeight()
{
	return weight;
}

double Edge::getFutureChange()
{
	return futureChange;
}

void Edge::setFutureChange(const double change)
{
	futureChange = change;
}

void Edge::updateWeight(const double change)
{
	weight -= change;
}

int Edge::getFirstVacantId()
{
	return edgesCnt;
}
