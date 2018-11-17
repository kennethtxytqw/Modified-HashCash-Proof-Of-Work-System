#include "ProofOfWorkGenerator.cuh"
#include <chrono>
#include <iostream>
#include <limits>
#include "hash.cuh"

using namespace ProofOfWork;
using namespace std::chrono;

void check_cuda_errors()
{
    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
    {
        cerr << "Last CUDA error " << cudaGetErrorString(rc) << endl;
    }

}

__global__ void generateKernel(const uint8_t* templateX, ullong* nonce, uint8_t* digest, ulong target, int* found)
{
    unsigned long long perThread = ULLONG_MAX / gridDim.x / gridDim.y / blockDim.x / blockDim.y;
    unsigned long long tid =  ((gridDim.x * gridDim.y) * blockIdx.z + (gridDim.x) * blockIdx.y + blockIdx.x) * (blockDim.x * blockDim.y) + blockDim.x * threadIdx.y + threadIdx.x;
    
    ullong start = tid * perThread;
    ullong stop = start + perThread;

    // For checking if the thread are checking the correct range
    // printf("%llu: %llu -> %llu:%llu\n", tid, start, stop, target);

    uint8_t candidateX[X_SIZE/BYTE_SIZE];

    // Copy templateX in, templateX is in little-endian, candidateX is in big-endian
    unsigned len = (X_SIZE-NONCE_SIZE)/BYTE_SIZE;
    for(unsigned i=0; i<len; ++i)
    {
        candidateX[len - i - 1] = templateX[i];
    }

    for(ullong i=start; i < stop; ++i)
    {
        if((*found) > 0)
        {
            return;
        } else 
        {
            ullong_to_uint8_big_endian(candidateX + len, i);
            // printf("%llu: Trying %llu...\n", tid, i);
            uint8_t hash[DIGEST_SIZE_IN_BYTES];
            sha256(hash, candidateX, X_SIZE);
            bool verified = verify(hash, target);

            if(verified && !atomicCAS(found, 0, verified))
            {
                atomicExch(nonce, i);
                for(unsigned j=0; j<DIGEST_SIZE_IN_BYTES;++j)
                {
                    digest[j] = hash[j];
                }
                // printf("%llu found %llu\n", tid, i);
                return;
            }
        }
    }
}

__device__ bool verify(const uint8_t hash[32], ulong target)
{
    return uint8_big_endian_to_ulong(hash) < target;
}

ProofOfWorkGenerator::ProofOfWorkGenerator(const string prevDigest, const string id, ulong target, unsigned blockDimX, unsigned blockDimY, unsigned gridDimX, unsigned gridDimY):
target(target)
{
    if (id.length() > ID_SIZE/HEX_SIZE)
    {
        cerr << "id needs to be of length 1-" << ID_SIZE/HEX_SIZE << ", id received: " << id.c_str() << endl;
        exit(1);
    }

    if (prevDigest.length() > DIGEST_SIZE_IN_BITS/HEX_SIZE)
    {
        cerr << "digest needs to be of length " << DIGEST_SIZE_IN_BITS/HEX_SIZE << " in a hex string, digest received: " << prevDigest.c_str() << endl;
        exit(1);
    }
    
    this->id = id;
    cerr << "Id: " << this->id << endl;
    this->prevDigest = prevDigest;
    cerr << "Prev Digest: " << this->prevDigest << endl;
    this->epoch = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
    cerr << "Epoch: " << this->epoch << endl;
    this->target = target;
    cerr << "Target: " << this->target << endl;

    this->blockDimX = blockDimX;
    this->blockDimY = blockDimY;
    this->gridDimX = gridDimX;
    this->gridDimY = gridDimY;

    cudaMallocManaged(&(this->templateX), (X_SIZE-NONCE_SIZE)/BYTE_SIZE);
    cudaMallocManaged(&(this->nonce), sizeof(ullong));
    cudaMallocManaged(&(this->targetUint8), TARGET_SIZE/BYTE_SIZE);
    cudaMallocManaged(&(this->found), sizeof(int));
    cudaMallocManaged(&(this->digest), DIGEST_SIZE_IN_BYTES);
    this->found[0] = 0;
    this->init();

    cerr << "Instantiated ProofOfWorkGenerator\n";
}

ProofOfWorkGenerator::~ProofOfWorkGenerator(void)
{
    cudaFree(this->templateX);
    cudaFree(this->nonce);
    cudaFree(this->targetUint8);
    cudaFree(this->found);
}

void ProofOfWorkGenerator::generate()
{
    dim3 blockDim(this->blockDimX, this->blockDimY);
    dim3 gridDim(this->gridDimX, this->gridDimY);

    generateKernel<<<gridDim, blockDim>>>(this->templateX, this->nonce, this->digest, this->target, this->found);
    cudaDeviceSynchronize();

    if(*(this->found) > 0)
    {
        cerr << "Found " << this->getNonce() << endl;
    } else 
    {
        cerr << "Failed\n";
    }
    check_cuda_errors();
}

string ProofOfWorkGenerator::getId()
{
    return this->id;
}

ulong ProofOfWorkGenerator::getEpoch()
{
    return this->epoch;
}

ullong ProofOfWorkGenerator::getNonce()
{
    return *(this->nonce);
}

string ProofOfWorkGenerator::getDigest()
{
    return uint8_to_hexstring(this->digest, DIGEST_SIZE_IN_BYTES);
}

string ProofOfWorkGenerator::getTemplateXHexString()
{
    return uint8_to_hexstring(this->templateX, (X_SIZE-NONCE_SIZE)/BYTE_SIZE);
}

void ProofOfWorkGenerator::init()
{
    cerr << "Initial templateX:\n" << this->getTemplateXHexString() << "\n\n";

    ascii_str_to_uint8_little_endian(this->templateX, this->id);
    cerr << "After inserting id:\n" << this->getTemplateXHexString() << "\n\n";

    hex_str_to_uint8_little_endian(this->templateX + (ID_SIZE)/BYTE_SIZE, this->prevDigest);
    cerr << "After inserting prev digest:\n" << this->getTemplateXHexString() << "\n\n";
    
    ulong_to_uint8_little_endian(this->templateX + (ID_SIZE + DIGEST_SIZE_IN_BITS)/BYTE_SIZE, this->epoch);
    cerr << "After inserting epoch:\n" << this->getTemplateXHexString() << "\n\n";

    ulong_to_uint8_little_endian(this->targetUint8, this->target);

}

