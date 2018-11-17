#include "ProofOfWorkGenerator.cuh"
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>

using namespace ProofOfWork;
using namespace std::chrono;
using namespace std;

int main(int argc, char **argv)
{
    auto start = system_clock::now();
    duration<double> duration;

    if (argc != 6) {
        cerr << "Usage: program <inputfile> <blockDimX> <blockDimY> <gridDimX> <gridDimY>\n"; 
        exit(1);
    }

    unsigned blockDimX, blockDimY, gridDimX, gridDimY;
    blockDimX = stoi(string(argv[2]));
    blockDimY = stoi(string(argv[3]));
    gridDimX = stoi(string(argv[4]));
    gridDimY = stoi(string(argv[5]));

    ifstream infile(argv[1]);
    string line;
    string initialDigest;
    string id = "E0011845";
    ulong target;

    getline(infile, line);
    initialDigest = line;
    getline(infile, line);
    target = stoul(line);

    ProofOfWorkGenerator* gen = new ProofOfWorkGenerator(initialDigest, id, target, blockDimX, blockDimY, gridDimX, gridDimY);
    duration = (system_clock::now() - start);
    cerr << "Instantiation time: " << duration.count() << " seconds." << endl;

    auto generate_start = system_clock::now();
    gen->generate();
    duration = (system_clock::now() - generate_start);
    cerr << "Generation time: " << duration.count() << " seconds." << endl;

    duration = (system_clock::now() - start);
    cerr << "Total time: " << duration.count() << " seconds." << endl;

    cout << gen->getId() << endl << gen->getEpoch() << endl << gen->getNonce() << endl << gen->getDigest() << endl;
    return 0;
}