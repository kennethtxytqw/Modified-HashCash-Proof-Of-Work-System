CC=nvcc
CFLAGS= -std=c++11 -I. -arch=sm_30
prog_name= ProofOfWorkGenerator

all: $(prog_name)
.PHONY: benchmark

ProofOfWorkGenerator.o: ProofOfWorkGenerator.cu 
	$(CC) $(CFLAGS) -dc $^

hash.o: hash.cu 
	$(CC) $(CFLAGS) -dc $^

main.o: main.cu 
	$(CC) $(CFLAGS) -dc $^

utils.o: utils.cu 
	$(CC) $(CFLAGS) -dc $^

$(prog_name): ProofOfWorkGenerator.o hash.o main.o utils.o
	$(CC) $(CFLAGS) $^ -o $@

benchmark: $(prog_name)
	mkdir -p outputs/
	./$(prog_name) inputs/1.in 1 1 1 1 1> outputs/1.out 2> outputs/1.err
	./$(prog_name) inputs/2.in 1 1 1 1 1> outputs/2.out 2> outputs/2.err
	./$(prog_name) inputs/3.in 1 1 1 1 1> outputs/3.out 2> outputs/3.err

	
	./$(prog_name) inputs/4.in 1 1 1 1 1> outputs/4.out 2> outputs/4.err
	./$(prog_name) inputs/5.in 1 1 1 1 1> outputs/5.out 2> outputs/5.err

clean:
	rm -rf $(prog_name) ProofOfWorkGenerator.o hash.o main.o utils.o outputs/

