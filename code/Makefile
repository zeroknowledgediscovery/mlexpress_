CC = g++ 
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Wunused -fopenmp -Wl,--as-needed
CFLAGS_DYN = -std=c++11  -O3 -Wall -Wextra -Wunused -fopenmp -Wl,--as-needed
ZBASE=./

LIBSO=  -lgomp  -lm 
LIBS=   -lboost_system -lboost_thread -lboost_program_options -lboost_timer  -lboost_chrono -lgsl -lgslcblas
LIBPATH= $(ZBASE)/lib

DEPS = 

INCLUDES = -I$(ZBASE)

OBJ =   makesample  makesample2

all:	$(OBJ)    clear mvbin


# compile libraries --------------------------------
semantic.o: semantic.cc $(DEPS)
	$(CC) -c -o $@ $< -fopenmp $(CFLAGS) $(INCLUDES); ar rcs libsemcrct.a semantic.o; rm semantic.o; mv libsemcrct.a ./lib

config.o: config.cc $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS); ar rcs libconfigfile.a config.o; rm config.o; mv libconfigfile.a ./lib

# compile binaries --------------------------------


makesample.o: makesample.cc
	$(CC) $(INCLUDES)  -c -o $@ $< $(CFLAGS)

makesample: makesample.o
	$(CC)  $(CFLAGS) -o $@ $^  $(LIBS) 



makesample2.o: makesample2.cc
	$(CC) $(INCLUDES)  -c -o $@ $< $(CFLAGS)

makesample2: makesample2.o
	$(CC)  $(CFLAGS) -o $@ $^  $(LIBS) 



# utility commands --------------------------------

clear:	
	rm -rf *.o
clean:	
	rm -rf *.o *~ $(OBJ) *.tgz

mvbin:
	rm -rf ./bin >&/dev/null; mkdir bin; mv $(OBJ) ./bin
src:
	tar -czvf src.tgz *.cc *.h Makefile Doxyfile
ref:
	doxygen Doxyfile; cd doc/latex; make; make

 # EOF --------------------------------
