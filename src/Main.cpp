#include "NeuralNetwork.h"
#include <string>

int main()
{
	string operation;
	cout << "Enter boolean operation:\n";
	cout << "Supported operations: xor, and, or.\n";
	cin >> operation;

	vector<pair<double, double>> inputValue = { {0,0},{0,1},{1,0},{1,1} };

	vector<double> expectedOutput; 
	if (operation == "xor")
	{
		expectedOutput = { 0,1,1,0 };
	}
	else if (operation == "and")
	{
		expectedOutput = { 0,0,0,1 };
	}
	else if (operation == "or")
	{
		expectedOutput = { 0,1,1,1 };
	}
	else
	{
		cout << "Operation not recognized!\n";
		return 0;
	}

	cout << "                   Neural Network 2-2-1\n";
	cout << "    Error      |                          Output\n";
	NeuralNetwork n1(inputValue, expectedOutput);
	n1.train();
	
	cout << "\n\n\n";
	cout << "                   Neural Network 2-4-1\n";
	cout << "    Error      |                          Output\n";
	NeuralNetwork n2(inputValue, expectedOutput, 2, 4, 1);
	n2.train();

	return 0;
}