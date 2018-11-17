# Usage
```
int main(int argc, char **arv)
{
    ProofOfWorkGenerator* gen = new ProofOfWorkGenerator(string("PREV DIGEST"), string("ID"), target);
    gen->generate();
    return 0;
}
```