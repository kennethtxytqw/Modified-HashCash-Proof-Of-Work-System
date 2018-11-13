#pragma once
#include <string>
#include <bitset>

#define X_SIZE 416 // bits
#define ID_SIZE 64 // bits
#define EPOCH_SIZE 32 // bits
#define NONCE_SIZE 64 // bits
#define DIGEST_SIZE 256 // bits

using namespace std;
namespace ProofOfWork 
{
    class ProofOfWorkGenerator
    {
        public:
            ProofOfWorkGenerator(string prevDigest, string id, int target);
            ~ProofOfWorkGenerator();

            void generate();

            ulong getEpoch();
            ulong getNonce();
            uint8_t* getDigest();
            string getId();
        
        private:
            bitset<DIGEST_SIZE> prevDigest;
            bitset<ID_SIZE> id;
            bitset<EPOCH_SIZE> epoch;
            bitset<X_SIZE> result;
            bitset<NONCE_SIZE> nonce;
            uint8_t digest[32];

            int target;
    };
}