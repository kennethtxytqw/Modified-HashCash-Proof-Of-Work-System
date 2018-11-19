CC=nvcc
CFLAGS= -std=c++11 -I. -arch=sm_52
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
	mkdir -p outputs/ outputs/sp/
	@for expnum in 1 2 3 4 5 ; do \
		for threadnum in 256 128 64 32 ; do \
			for timeth in 1 2 3 4 5 6 7 8 9 10; do \
				./$(prog_name) inputs/$$expnum.in 64 1 $$threadnum 1 1> outputs/sp/$$expnum-$$threadnum.out$$timeth 2> outputs/sp/$$expnum-$$threadnum.err$$timeth ; \
			done \
		done \
	done

	@for expnum in 1 2 ; do \
		for threadnum in 1 ; do \
			for timeth in 1 2 3 4 5 6 7 8 9 10; do \
				./$(prog_name) inputs/$$expnum.in 64 1 $$threadnum 1 1> outputs/sp/$$expnum-$$threadnum.out$$timeth 2> outputs/sp/$$expnum-$$threadnum.err$$timeth ; \
			done \
		done \
	done

	python3 collate.py outputs/sp/

clean:
	rm -rf $(prog_name) ProofOfWorkGenerator.o hash.o main.o utils.o

