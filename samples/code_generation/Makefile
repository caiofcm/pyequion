

# Remove -g if not debug mode
CFLAGS := -Wall -std=c++11 -O3 -march=native -ffp-contract=fast -g

all: libcalc_res.so solve_eq

# $@ matches the target; $< matches the first dependent
libcalc_res.so: calc_cnv_res_equilibrium_NaHCO3_CaCl2.o
	gcc -shared -o $@ $< -lm

calc_cnv_res_equilibrium_NaHCO3_CaCl2.o: calc_cnv_res_equilibrium_NaHCO3_CaCl2.c
	gcc -o $@ -c $<

# main_test_exported: main_test_exported.c
# 	gcc $< -L  ./ -lcalc_res -o main_test_exported
main_test_exported: main_test_exported.cpp
	g++ $< -L  ./ -lcalc_res -o main_test_exported

solve_eq: solve_eq.cpp
	# g++ -I/usr/local/include/optim/ solve_eq.cpp -o optim_de_ex.out  -loptim
	g++ $(CFLAGS) -I/usr/local/include/optim/ solve_eq.cpp -o optim_de_ex.out  -loptim -L  ./ -lcalc_res

clean:
	rm calc_cnv_res_equilibrium_NaHCO3_CaCl2.o libcalc_res.so main_test_exported


# Reference on Makefile: https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html
# Reference on Shared Libraries: https://www.cs.swarthmore.edu/~newhall/unixhelp/howto_C_libraries.html

# Usage:
# export LD_LIBRARY_PATH=/home/caio/Projects/CarbonateDeposition/pyequion/samples/code_generation:$LD_LIBRARY_PATH
