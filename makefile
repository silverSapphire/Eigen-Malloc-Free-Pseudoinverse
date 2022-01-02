run: demo
	./demo

clean:
	rm *.o demo

# Linking
demo: demo.o PINV.o
	g++ -o demo PINV.o demo.o

# Compilation
demo.o: demo.cpp
	g++ -c -O2 demo.cpp
PINV.o: PINV.h PINV.cpp
	g++ -c -O2 PINV.cpp
