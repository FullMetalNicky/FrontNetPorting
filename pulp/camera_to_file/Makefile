PULP_APP = test
PULP_APP_FC_SRCS = test.c ImgIO.c
PULP_CFLAGS = -O3 -g #-DFPGA

# This tells the platform to activate what is needed to connect the debug bridge
bridge=1
override CONFIG_OPT += camera/image-stream=$(CURDIR)/index.png
override CONFIG_OPT += runner/peripherals=true
#PLPBRIDGE_FLAGS += -fileIO 5

ifdef RAW
override CONFIG_OPT += camera/color-mode=raw
PULP_CFLAGS += -DUSE_RAW=1
endif

clean::
	rm -rf *.ppm

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
