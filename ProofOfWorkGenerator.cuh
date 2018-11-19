#pragma once
#include <string>
#include "utils.cuh"

typedef unsigned long long int ullong;

__global__ void generateKernel(const uint8_t* templateX, ullong* nonce, uint8_t* digest, ulong target, int* found);
__device__ bool verify(const uint8_t hash[32], ulong target);

using namespace std;
namespace ProofOfWork 
{
    class ProofOfWorkGenerator
    {
        public:
            ProofOfWorkGenerator(const std::string prevDigest, const std::string id, const ulong target, unsigned gridDimX, unsigned gridDimY, unsigned blockDimX, unsigned blockDimY);
            ~ProofOfWorkGenerator();

            void generateCudaDeviceSynchronize();
            void generateBusyWait();
            void generateBusyWaitWithRand();

            ulong getEpoch();
            ullong getNonce();
            std::string getId();
            std::string getTemplateXHexString();
            std::string getDigest();
        
        private:
            std::string prevDigest;
            std::string id;
            ulong epoch;
            ulong target;
            uint8_t* templateX; // little endian
            uint8_t* targetUint8;
            ullong* nonce; // little endian
            int* found;

            unsigned gridDimX;
            unsigned gridDimY;

            unsigned blockDimX;
            unsigned blockDimY;

            uint8_t* digest; // little endian
            void init();
    };
}