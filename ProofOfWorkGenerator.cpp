#include "ProofOfWorkGenerator.h"
#include <chrono>
#include "hash.h"

using namespace ProofOfWork;
using namespace std::chrono;

ProofOfWorkGenerator::ProofOfWorkGenerator(string prevDigest, string id, int target):
target(target)
{
    this->id = bitset<ID_SIZE>(id);
    this->prevDigest = bitset<DIGEST_SIZE>(prevDigest);
    this->epoch = bitset<EPOCH_SIZE>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
}

ProofOfWorkGenerator::~ProofOfWorkGenerator(void)
{
    free(&(this->id));
    free(&(this->prevDigest));
}

void ProofOfWorkGenerator::generate()
{
    this->nonce = bitset<NONCE_SIZE>("Magic Nonce");
    this->result = bitset<EPOCH_SIZE + DIGEST_SIZE + ID_SIZE + NONCE_SIZE>(
        this->epoch.to_string + 
        this->prevDigest.to_string + 
        this->id.to_string + 
        this->nonce.to_string);

    sha256(this->digest, this->result, X_SIZE);
}

string ProofOfWorkGenerator::getId()
{
    return this->id.to_string();
}

ulong ProofOfWorkGenerator::getEpoch()
{
    return this->epoch.to_ullong();
}

uint8_t* ProofOfWorkGenerator::getDigest()
{
    return this->digest;
}

ulong ProofOfWorkGenerator::getNonce()
{
    return this->nonce.to_ulong;
}