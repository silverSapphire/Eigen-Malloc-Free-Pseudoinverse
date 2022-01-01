run: demo
	./demo

# Linking
demo: demo.o PINV.o
	g++ -o demo PINV.o demo.o

# Compilation
demo.o: demo.cpp
	g++ -c demo.cpp
PINV.o: PINV.h PINV.cpp
	g++ -c PINV.cpp
