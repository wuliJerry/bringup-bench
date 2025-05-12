
LOCAL_CFLAGS=
LOCAL_LIBS=
LOCAL_OBJS=mnist-nn.o soft_float_ops.o

PROG=mnist-nn

include ../Makefile

clean_float:
	rm -f soft_float_ops.o

test: $(TARGET_EXE)
	-$(TARGET_SIM) ./$(TARGET_EXE) > FOO 
	$(TARGET_DIFF) FOO mnist-nn.$(TARGET_REFEXT)
	rm -f FOO

