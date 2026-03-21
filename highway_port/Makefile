CXX      = riscv64-unknown-elf-g++
CXXFLAGS = -O2 -std=c++17 -I. -I../highway -march=rv64gcv -mabi=lp64d
QEMU     = qemu-riscv64 -cpu rv64,v=on,vext_spec=v1.0,vlen=128,rvv_ta_all_1s=on

HWY_SRCS = ../highway/hwy/abort.cc \
           ../highway/hwy/aligned_allocator.cc \
           ../highway/hwy/per_target.cc \
           ../highway/hwy/targets.cc \
           ../highway/hwy/print.cc

HWY_OBJS = $(HWY_SRCS:.cc=.o)

all: rvv_example

rvv_example: ssw_hwy.o example.o $(HWY_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

ssw_hwy.o: ssw_hwy.cpp ssw.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

example.o: example.c ssw.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

../highway/hwy/%.o: ../highway/hwy/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: rvv_example
	$(QEMU) ./rvv_example

clean:
	rm -f ssw_hwy.o example.o $(HWY_OBJS) rvv_example

.PHONY: all run clean
